"""SplitSnap core logic (framework-free, so it can be tested without a web server).

  python app.py                                   # mock extractor (UI work, no model)
  SPLITSNAP_CKPT=/path/to/ckpt_best python app.py # real Donut checkpoint from the notebook
  SPLITSNAP_REMOTE=https://xyz.ngrok.app python app.py   # forward scans to a model served elsewhere
"""
import io, json, os, re, sys, threading
from splitsnap_engine import *
from splitsnap_engine.bill import charge_category

CKPT, REMOTE = os.getenv("SPLITSNAP_CKPT"), os.getenv("SPLITSNAP_REMOTE")
MODE = "donut" if CKPT else "remote" if REMOTE else "mock"

SAMPLE = {"items": [{"name": "Paneer Butter Masala", "qty": 1, "unit_price": 220, "total": 220},
                    {"name": "Garlic Naan", "qty": 3, "unit_price": 40, "total": 120},
                    {"name": "Veg Biryani", "qty": 1, "unit_price": 260, "total": 260},
                    {"name": "Cold Coffee", "qty": 1, "unit_price": 140, "total": 140}],
          "charges": [{"type": "CGST", "amount": 18.5}, {"type": "SGST", "amount": 18.5},
                      {"type": "Service Charge", "amount": 40}, {"type": "Discount", "amount": -50}],
          "subtotal": 740, "grand_total": 767}
SAMPLE_CONF = {**{f"items[{i}].{k}": {"min": 0.97} for i in range(4) for k in ("name", "qty", "total")},
               **{f"charges[{i}].amount": {"min": 0.96} for i in range(4)},
               "items[3].total": {"min": 0.62}, "charges[3].amount": {"min": 0.71}, "items[1].name": {"min": 0.76}}


def _num(s):
    """'Rs.7.75' -> 7.75, '₹1,234.50' -> 1234.5, '1.234' -> 1234, '-50' -> -50."""
    if s is None: return None
    s = str(s).strip()
    m = re.search(r"\d[\d.,]*", s)
    if not m: return None
    neg = "-" in s[:m.start()] or s.startswith("(")
    t = m.group().rstrip(".,")
    if "." in t and "," in t:
        t = t.replace(",", "") if t.rfind(".") > t.rfind(",") else t.replace(".", "").replace(",", ".")
    else:
        for sep in ".,":
            if sep in t:
                parts = t.split(sep)
                if len(parts) > 2 or len(parts[-1]) == 3: t = "".join(parts)       
                else: t = ".".join(parts)                                         
    try: v = float(t)
    except ValueError: return None
    v = -v if neg else v
    return int(v) if v == int(v) else v


def _field_conf(toks, probs):
    import numpy as np
    stack, bag = [], {}
    for t, p in zip(toks, probs):
        if re.fullmatch(r"<s_[^>]+>", t): stack.append([t[3:-1], 0])
        elif re.fullmatch(r"</s_[^>]+>", t):
            if stack: stack.pop()
        elif t == "<sep/>":
            if stack: stack[-1][1] += 1
        elif stack:
            path = ".".join([f"{k}[{i}]" for k, i in stack[:-1]] + [stack[-1][0]])
            bag.setdefault(path, []).append(p)
    return {k: {"min": float(min(v))} for k, v in bag.items()}


_M = {}
_LOCK = threading.Lock()         


def _token2json(tokens, added=frozenset(), inner=False):
    """Donut's own sequence -> dict parser (same algorithm as DonutProcessor.token2json)."""
    out = {}
    while tokens:
        st = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if st is None: break
        key = st.group(1)
        en = re.search(rf"</s_{re.escape(key)}>", tokens, re.IGNORECASE)
        st = st.group()
        if en is None:
            tokens = tokens.replace(st, "")
            continue
        en = en.group()
        m = re.search(f"{re.escape(st)}(.*?){re.escape(en)}", tokens, re.IGNORECASE | re.DOTALL)
        if m is not None:
            content = m.group(1).strip()
            if "<s_" in content and "</s_" in content:
                v = _token2json(content, added, True)
                if v:
                    out[key] = v[0] if len(v) == 1 else v
            else:
                leaves = []
                for leaf in content.split("<sep/>"):
                    leaf = leaf.strip()
                    if leaf in added and leaf[0] == "<" and leaf[-2:] == "/>": leaf = leaf[1:-2]
                    leaves.append(leaf)
                out[key] = leaves[0] if len(leaves) == 1 else leaves
        tokens = tokens[tokens.find(en) + len(en):].strip()
        if tokens[:6] == "<sep/>":
            return [out] + _token2json(tokens[6:], added, True)
    return ([out] if inner else out) if out else ([] if inner else {"text_sequence": tokens})


def _preprocess(img, h, w):
    """Donut's training-time image pipeline, in plain PIL/numpy: resize short edge, shrink to fit, centre-pad black, normalise."""
    import numpy as np
    from PIL import Image
    iw, ih = img.size
    short = min(h, w)
    nw, nh = (short, int(short * ih / iw)) if iw <= ih else (int(short * iw / ih), short)
    img = img.resize((nw, nh), Image.BICUBIC)
    th, tw = min(nh, h), min(nw, w)
    if (th, tw) != (nh, nw):
        if nh > nw: tw = int(nw * th / nh)
        elif nw > nh: th = int(nh * tw / nw)
        img = img.resize((max(1, tw), max(1, th)), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32)
    ph, pw = h - a.shape[0], w - a.shape[1]
    pt, pl = ph // 2, pw // 2
    a = np.pad(a, ((pt, ph - pt), (pl, pw - pl), (0, 0)))
    return ((a / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]


_M = {}
_LOCK = threading.Lock()       


def _donut(img):
    with _LOCK:
        return _donut_impl(img)


def _donut_impl(img):
    import time
    import torch
    from tokenizers import Tokenizer
    from transformers import VisionEncoderDecoderModel
    if not _M:
        t0 = time.time()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(os.cpu_count() or 4)
        try:
            model = VisionEncoderDecoderModel.from_pretrained(CKPT, low_cpu_mem_usage=False)
        except TypeError:
            model = VisionEncoderDecoderModel.from_pretrained(CKPT)
        model = model.eval().to(dev)
        if dev == "cpu" and os.getenv("SPLITSNAP_FAST") == "1":      
            try:
                model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                print("[donut] int8 quantisation on", flush=True, file=sys.stderr)
            except Exception as e:
                print("[donut] quantisation skipped:", e, flush=True, file=sys.stderr)
        tk = Tokenizer.from_file(os.path.join(CKPT, "tokenizer.json"))   # no transformers tokenizer classes involved
        gc = model.generation_config
        ids = {n: (tk.token_to_id(t) if tk.token_to_id(t) is not None else d)
               for n, t, d in (("pad", "<pad>", gc.pad_token_id), ("eos", "</s>", gc.eos_token_id), ("unk", "<unk>", None))}
        try: added = {t.content for t in tk.get_added_tokens_decoder().values()}
        except Exception: added = set()
        _M.update(tk=tk, ids=ids, added=added, dev=dev, model=model,
                  cfg=json.load(open(os.path.join(CKPT, "splitsnap_config.json"))))
        print(f"[donut] model loaded on {dev} in {time.time()-t0:.0f}s, {torch.get_num_threads()} threads", flush=True, file=sys.stderr)
    t_scan = time.time()
    tk, ids, model, dev, cfg = _M["tk"], _M["ids"], _M["model"], _M["dev"], _M["cfg"]
    h, w = (int(v) for v in os.getenv("SPLITSNAP_SIZE", f"{cfg['img_h']}x{cfg['img_w']}").split("x"))
    img.thumbnail((2400, 2400))
    pv = torch.from_numpy(_preprocess(img, h, w)).to(dev)
    dec = torch.tensor([[tk.token_to_id(cfg["tasks"]["bill"])]], device=dev)
    kw = {"bad_words_ids": [[ids["unk"]]]} if ids["unk"] is not None else {}
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
        out = model.generate(pv, decoder_input_ids=dec, max_length=cfg["max_len"], pad_token_id=ids["pad"],
                             eos_token_id=ids["eos"], use_cache=True, num_beams=1, do_sample=False,
                             return_dict_in_generate=True, output_scores=True, **kw)
    lp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
    probs, gen = lp.float().exp().tolist(), out.sequences[0][1:].tolist()
    gen = gen[:len(probs)]
    cut = gen.index(ids["eos"]) if ids["eos"] in gen else len(gen)
    gen, probs = gen[:cut], probs[:cut]
    toks = [tk.id_to_token(i) or "" for i in gen]
    seq = tk.decode(gen, skip_special_tokens=False)
    js = _token2json(seq, _M["added"])
    print(f"[donut] {time.time()-t_scan:.1f}s for {len(gen)} tokens at {h}x{w}\n[donut] raw: {seq[:500]}\n[donut] parsed keys: {list(js) if isinstance(js, dict) else js}", flush=True, file=sys.stderr)
    lst = lambda x: x if isinstance(x, list) else ([x] if isinstance(x, dict) else [])
    raw = {"items": [{"name": str(i.get("name", "")).strip(), "qty": _num(i.get("qty")),
                      "unit_price": _num(i.get("unit_price")), "total": _num(i.get("total"))}
                     for i in lst(js.get("items")) if isinstance(i, dict)],
           "charges": [{"type": str(c.get("type", "")).strip(), "amount": _num(c.get("amount"))}
                       for c in lst(js.get("charges")) if isinstance(c, dict)],
           "subtotal": _num(js.get("subtotal")), "grand_total": _num(js.get("grand_total"))}
    return raw, _field_conf(toks, probs)


def _open_image(data):
    """Open whatever the user uploaded: honour phone rotation, flatten transparency onto white, explain unsupported files."""
    from PIL import Image, ImageOps
    try:
        from pillow_heif import register_heif_opener     
        register_heif_opener()
    except Exception:
        pass
    try:
        img = Image.open(io.BytesIO(data)); img.load()
    except Exception as e:
        raise SplitError("This file isn't an image the app can open. JPG, PNG and WebP work. If it is a PDF, HEIC or AVIF file, "
                         "convert it to JPG/PNG first or take a screenshot of the bill.") from e
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")                          
        bg = Image.new("RGBA", img.size, "white"); bg.alpha_composite(img); img = bg
    elif img.mode in ("I;16", "I", "F"):
        img = img.point(lambda v: v / 256).convert("L") if img.mode == "I;16" else img.convert("L")
    return img.convert("RGB")


def extract(data):
    if MODE == "mock":
        return SAMPLE, SAMPLE_CONF
    if MODE == "remote":
        import urllib.request
        rq = urllib.request.Request(REMOTE.rstrip("/") + "/extract", data=data,
                                    headers={"Content-Type": "application/octet-stream", "User-Agent": "splitsnap/1.0"})
        j = json.load(urllib.request.urlopen(rq, timeout=120))
        return j["pred"], j.get("conf") or {}
    return _donut(_open_image(data))



def warmup():
    """Load the model at server start so the first real scan doesn't pay for it."""
    if MODE == "donut":
        try:
            from PIL import Image
            _donut(Image.new("RGB", (800, 1000), "white"))
            print("[donut] ready for scans", flush=True, file=sys.stderr)
        except Exception as e:
            print("[donut] warm-up failed:", e, flush=True, file=sys.stderr)


_META = re.compile(r"^\s*(date|time|bill|invoice|inv|table|tbl|order|token|cashier|server|waiter|steward|gstin?|fssai|phone|ph|tel|mob|covers?|pax|kot|ref|dine\s*in)\b\s*(?:[:#.]|no\b|\d|$)"
                   r"|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\b\d{1,2}:\d{2}\b", re.I)


def _clean_raw(raw, conf):
    """Guards against the model's known habits: header lines read as items, and dropped decimal points (90.00 -> 9000)."""
    notes, keep, items, fixed = [], [], [], 0
    for i, it in enumerate(raw.get("items") or []):
        if _META.search(str(it.get("name") or "")):
            notes.append(f"Ignored '{it['name']}': it looks like a header line, not an item.")
            continue
        u, t, q = it.get("unit_price"), it.get("total"), it.get("qty")
        if u and t and t > 0 and u > t * 1.0001:          
            it = dict(it, unit_price=round(t / (q or 1), 2)); fixed += 1
        keep.append(i); items.append(it)
    if fixed:
        notes.append(f"Corrected {fixed} unit price(s) that were bigger than the line total (probably a misread decimal point). Please check them.")
    where = {old: new for new, old in enumerate(keep)}
    conf2 = {}
    for k, v in conf.items():
        m = re.match(r"items\[(\d+)\](.*)", k)
        if not m: conf2[k] = v
        elif int(m.group(1)) in where: conf2[f"items[{where[int(m.group(1))]}]{m.group(2)}"] = v
    return dict(raw, items=items), conf2, notes


def scan_bill(data, sample=False):
    raw, conf = (SAMPLE, SAMPLE_CONF) if sample else extract(data)
    raw, conf, notes = _clean_raw(raw, conf)
    bill, warnings, flags = normalize_bill(raw, conf)
    warnings = notes + warnings
    ratio = len(flags) / max(1, len(conf))
    retake = (not bill["items"]) or ratio > 0.6          
    if not bill["items"]:
        warnings.append("No items could be read from this photo.")
    return {"bill": bill, "warnings": warnings, "flags": flags, "extractor": MODE, "retake": retake, "uncertain": len(flags), "fields": len(conf),
            "summary": summarize_charges(bill)}


def _prep(b):
    """UI edits arrive as plain rows; make them engine-clean without losing ids."""
    raw = {"items": b.get("items") or [], "charges": b.get("charges") or [], "grand_total": b.get("stated_total")}
    bill, warns, _ = normalize_bill(raw)
    for c in bill["charges"]:                        
        c["category"] = charge_category(c["type"], to_paise(c["amount"]))
    bill["stated_total"] = rupees(to_paise(raw["grand_total"])) if raw["grand_total"] not in (None, "") else None
    return bill, warns


def run_split(j):
    bill, warns = _prep(j.get("bill") or {})
    if not bill["items"]:
        raise SplitError("Add at least one item.")
    ids = {i["id"] for i in bill["items"]}
    res = split_bill(bill, j.get("party") or [], j.get("assignments") or {}, on_unassigned=j.get("on_unassigned", "report"),
                     manual_items=[i for i in j.get("manual_items") or [] if i in ids])
    res["warnings"] = warns + res["warnings"]
    for o in j.get("overrides") or []:                
        if o.get("name") in res["people"]:
            res = apply_total_override(res, o["name"], o["total"], o.get("reason", ""))
    out = {"result": res, "bill_text": explain_bill(res), "explain": explain_all(res),
           "explain_short": explain_all(res, "short"), "bill": bill}
    if j.get("paid_by"):
        out["settle"] = settle_up(res, j["paid_by"])
    return out