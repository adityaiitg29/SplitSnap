"""Run the fine-tuned SplitSnap Donut model on one bill image and print the extracted JSON.

    python infer.py path/to/bill.jpg                  # uses ./ckpt_best
    python infer.py path/to/bill.jpg --ckpt D:/models/ckpt_best

Output: items, charges, subtotal, grand_total, plus per-field confidence flags and any warnings.
Only open-source components are used: the Donut checkpoint fine-tuned in splitsnap_colab.ipynb (R4).
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("image")
ap.add_argument("--out", help="write the JSON to this file (UTF-8) instead of printing it")
ap.add_argument("--ckpt", default="ckpt_best", help="folder with model.safetensors, tokenizer.json, splitsnap_config.json")
a = ap.parse_args()
if not os.path.exists(os.path.join(a.ckpt, "splitsnap_config.json")):
    sys.exit(f"Checkpoint not found at '{a.ckpt}'. Download ckpt_best (see README) and pass --ckpt.")
os.environ["SPLITSNAP_CKPT"] = a.ckpt         
from core import scan_bill                      

out = scan_bill(open(a.image, "rb").read())
text = json.dumps({"bill": out["bill"], "low_confidence_fields": out["flags"], "warnings": out["warnings"],
                   "retake_suggested": out["retake"]}, indent=2, ensure_ascii=False)
if a.out:
    open(a.out, "w", encoding="utf-8").write(text)
    print(f"saved {a.out}")
else:
    print(text)