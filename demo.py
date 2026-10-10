"""Run: python demo.py   (uses the sample bill from the problem statement)"""
import json
from splitsnap_engine import *

raw = {"items": [{"name": "Paneer Butter Masala", "qty": 1, "unit_price": 220, "total": 220},
                 {"name": "Garlic Naan", "qty": 3, "unit_price": 40, "total": 120},
                 {"name": "Veg Biryani", "qty": 1, "unit_price": 260, "total": 260},
                 {"name": "Cold Coffee", "qty": 1, "unit_price": 140, "total": 140}],
       "charges": [{"type": "CGST", "amount": 18.5}, {"type": "SGST", "amount": 18.5},
                   {"type": "Service Charge", "amount": 40}, {"type": "Discount", "amount": -50}],
       "subtotal": 740, "grand_total": 767}

bill, warnings, flags = normalize_bill(raw)
party = ["Riya", "Aman", "Kabir"]
assign = {"i0": "Riya", "i1": {"Riya": 1, "Aman": 1, "Kabir": 1}, "i2": "Aman", "i3": "Kabir"}

res = split_bill(bill, party, assign)
print(explain_bill(res)); print()
for n in party:
    print(explain_person(res, n)); print()

print("--- Riya's amount set by hand to 280 ---")
res2 = apply_total_override(res, "Riya", 280, reason="she had an extra drink")
print(explain_bill(res2)); print(explain_person(res2, "Aman"))
print("\nAman paid the bill:", settle_up(res2, "Aman"))
