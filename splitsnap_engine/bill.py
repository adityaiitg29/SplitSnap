"""Turns raw extraction output (what the model notebook's `norm_pred` returns) into a clean
bill the split engine can trust, and reports everything that looked wrong.

Schema in (same as the training notebook):
    {"items":   [{"name", "qty", "unit_price", "total"}],
     "charges": [{"type", "amount"}],          # discounts negative
     "subtotal": .., "grand_total": ..}
"""
import re
from .money import to_paise, rupees, fmt_inr

_ROUND = ("round",)
_DISCOUNT = ("discount", "disc", "coupon", "promo", "offer", "voucher", "cashback", "saving")
_TAX = ("cgst", "sgst", "igst", "gst", "vat", "tax", "cess", "pb1")
_SERVICE = ("service", "svc", "serv chg", "s.charge", "s/c")
BASES = ("proportional", "equal")


def charge_category(ctype, amount_p=0):
    """tax | service | discount | rounding | other (order matters: 'Round Off' must not read as a discount)."""
    t = (ctype or "").lower()
    if any(w in t for w in _ROUND):
        return "rounding"
    if any(w in t for w in _DISCOUNT):
        return "discount"
    if any(w in t for w in _TAX):
        return "tax"
    if any(w in t for w in _SERVICE):
        return "service"
    return "discount" if amount_p < 0 else "other"


def _p(x):
    try:
        return to_paise(x)
    except ValueError:
        return None


def _qty(x):
    try:
        q = float(x)
    except (TypeError, ValueError):
        return 1
    if q <= 0 or q != q:
        return 1
    return int(q) if q.is_integer() else q


def normalize_bill(raw, conf=None, low_conf=0.80):
    """Returns (bill, warnings, flags).

    bill      clean bill; every item and charge has a stable id the UI can edit against
    warnings  human-readable problems found (mismatched totals, dropped lines, sign fixes)
    flags     fields whose model confidence is below `low_conf`, mapped to item/charge ids,
              so the correction screen can highlight exactly those cells (R2)
    """
    warnings, item_ids, charge_ids = [], {}, {}
    items, used = [], set()

    for ri, it in enumerate(raw.get("items") or []):
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()
        qty, unit, tot = _qty(it.get("qty")), _p(it.get("unit_price")), _p(it.get("total"))
        if not name:
            name = f"Item {ri + 1}"
            warnings.append(f"Line {ri + 1} had no name; called it '{name}'.")
        if tot is None and unit is None:
            warnings.append(f"Dropped '{name}': no price could be read.")
            continue
        if tot is None:
            tot = round(unit * qty)
        elif unit is None:
            unit = round(tot / qty)
        elif abs(round(unit * qty) - tot) > max(1, int(qty)):
            warnings.append(f"'{name}': {qty} × {fmt_inr(unit)} is not {fmt_inr(tot)}. Using the line total; check the price.")
        iid = str(it.get("id") or f"i{len(items)}")
        while iid in used:
            iid += "_"
        used.add(iid)
        item_ids[ri] = iid
        items.append({"id": iid, "name": name, "qty": qty, "unit_price": rupees(unit), "total": rupees(tot)})

    charges = []
    for ri, c in enumerate(raw.get("charges") or []):
        if not isinstance(c, dict):
            continue
        ctype = str(c.get("type") or "").strip() or "Other charge"
        amt = _p(c.get("amount"))
        if amt is None:
            warnings.append(f"Dropped charge '{ctype}': no amount could be read.")
            continue
        if amt == 0:
            continue
        cat = charge_category(ctype, amt)
        if cat == "discount" and amt > 0:
            amt = -amt
            warnings.append(f"'{ctype}' was positive; treated it as a deduction.")
        basis = c.get("basis", "proportional")
        if basis not in BASES:
            warnings.append(f"Unknown basis '{basis}' for '{ctype}'; splitting it proportionally.")
            basis = "proportional"
        cid = str(c.get("id") or f"c{len(charges)}")
        charge_ids[ri] = cid
        charges.append({"id": cid, "type": ctype, "category": cat, "amount": rupees(amt), "basis": basis})

    items_p = sum(to_paise(i["total"]) for i in items)
    stated_sub, stated_tot = _p(raw.get("subtotal")), _p(raw.get("grand_total"))
    if stated_sub is not None and abs(stated_sub - items_p) > 100:
        warnings.append(f"Items add up to {fmt_inr(items_p)} but the bill's subtotal says {fmt_inr(stated_sub)}. An item may be missing or misread.")

    flags = []
    for path, c in (conf or {}).items():
        score = c["min"] if isinstance(c, dict) else float(c)
        if score >= low_conf:
            continue
        m = re.fullmatch(r"(items|charges)\[(\d+)\]\.(\w+)", path)
        if m:
            ids = item_ids if m.group(1) == "items" else charge_ids
            if int(m.group(2)) in ids:
                flags.append({"target": m.group(1)[:-1], "id": ids[int(m.group(2))],
                              "field": m.group(3), "confidence": round(score, 3)})
        elif path in ("subtotal", "grand_total"):
            flags.append({"target": "bill", "id": None, "field": path, "confidence": round(score, 3)})
    flags.sort(key=lambda f: f["confidence"])

    bill = {"_normalized": True, "currency": "INR", "items": items, "charges": charges,
            "subtotal": rupees(items_p),
            "stated_subtotal": rupees(stated_sub) if stated_sub is not None else None,
            "stated_total": rupees(stated_tot) if stated_tot is not None else None}
    return bill, warnings, flags


def summarize_charges(bill):
    """The R5 display: Items, Taxes, Service Charge, Discount, Total, each shown separately."""
    items_p = sum(to_paise(i["total"]) for i in bill["items"])
    cats = {k: 0 for k in ("tax", "service", "discount", "rounding", "other")}
    by_type = []
    for c in bill["charges"]:
        a = to_paise(c["amount"])
        cats[c["category"]] += a
        by_type.append({"type": c["type"], "category": c["category"], "amount": rupees(a)})
    total = items_p + sum(cats.values())
    labels = [("tax", "Taxes"), ("service", "Service Charge"), ("discount", "Discount"),
              ("rounding", "Round Off"), ("other", "Other")]
    parts = [f"Items: {fmt_inr(items_p)}"] + [f"{lab}: {fmt_inr(cats[k])}" for k, lab in labels if cats[k]]
    parts.append(f"Total: {fmt_inr(total)}")
    return {"items": rupees(items_p), "taxes": rupees(cats["tax"]), "service": rupees(cats["service"]),
            "discount": rupees(cats["discount"]), "rounding": rupees(cats["rounding"]),
            "other": rupees(cats["other"]), "total": rupees(total), "by_type": by_type,
            "line": ", ".join(parts)}
