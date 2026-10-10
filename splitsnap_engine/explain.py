"""Plain-language explanations (R6). Built from the same numbers the split used, by template,
never by a model, so every figure in the text can be traced to the table."""
from fractions import Fraction

from .money import fmt_inr, to_paise


def _m(x):
    return fmt_inr(to_paise(x))


def _pct(x):
    return f"{x * 100:.1f}%"


def _share_text(fr):
    return f"{fr.numerator}/{fr.denominator}" if fr.denominator <= 12 else _pct(float(fr))


def _line_text(l):
    amt, fr = _m(l["amount"]), Fraction(l["share"])
    qty = int(l["qty"]) if float(l["qty"]).is_integer() else l["qty"]
    prefix = f"{qty} × " if qty != 1 else ""
    if fr == 1:
        text = f"{prefix}{l['name']} (full, {amt})"
    elif l["units"] is not None:
        text = f"{l['units']} of {qty} {l['name']} ({amt})"
    else:
        text = f"{_share_text(fr)} share of {prefix}{l['name']} ({amt})"
    if l["shared_with"]:
        text = text[:-1] + f", shared with {', '.join(l['shared_with'])})"
    if l["defaulted"]:
        text += " [split equally: nobody claimed it]"
    return text


def explain_person(result, name, style="detailed"):
    p = result["people"][name]
    sub, total = to_paise(p["items_subtotal"]), to_paise(p["total"])
    adj = p.get("adjustment")
    charges_net = to_paise(p["charges_total"])

    if style == "short":
        text = f"{name} owes {fmt_inr(total)}: items {fmt_inr(sub)} + share of taxes and charges {fmt_inr(charges_net)}."
        return text + (f" (manually adjusted from {_m(adj['original_total'])})" if adj else "")

    out = []
    if p["lines"]:
        out.append(f"{name}: " + " + ".join(_line_text(l) for l in p["lines"]) + f" = {fmt_inr(sub)} in items.")
    else:
        out.append(f"{name} has no items assigned.")

    prop = [c for c in p["charges"] if c["basis"] != "equal" and to_paise(c["amount"]) != 0]
    if prop and sub > 0:
        net = sum(to_paise(c["amount"]) for c in prop)
        pre = result["bill"]["items_total"]
        out.append(f"{name}'s items are {_pct(p['pretax_fraction'])} of the {_m(pre)} pre-tax bill, so {name} carries "
                   f"{_pct(p['pretax_fraction'])} of each shared charge: "
                   + ", ".join(f"{c['type']} {_m(c['amount'])}" for c in prop) + f" = {fmt_inr(net)} net.")
    for c in p["charges"]:
        if c["basis"] == "equal":
            out.append(f"{c['type']} ({_m(c['bill_amount'])}) is split equally among {len(result['party'])} people: {_m(c['amount'])}.")

    label = "Calculated total" if adj else "Final"
    out.append(f"{label}: {fmt_inr(sub)} {'+' if charges_net >= 0 else '-'} {fmt_inr(abs(charges_net))} = {fmt_inr(sub + charges_net)}."
               if charges_net else f"{label}: {fmt_inr(sub)}.")
    if adj and adj["type"] == "override":
        out.append(f"MANUAL ADJUSTMENT: set by hand to {fmt_inr(total)} (calculated {_m(adj['original_total'])})."
                   + (f" Reason: {adj['reason']}." if adj["reason"] else ""))
    elif adj:
        d = to_paise(adj["delta"])
        out.append(f"MANUAL ADJUSTMENT: {fmt_inr(abs(d))} {'more' if d > 0 else 'less'} than calculated because "
                   f"{adj['caused_by']}'s amount was set by hand. Final: {fmt_inr(total)}.")
    return "\n".join(out)


def explain_all(result, style="detailed"):
    return {n: explain_person(result, n, style) for n in result["people"]}


def explain_bill(result):
    s = result["summary"]
    out = [s["line"], ""]
    for n, p in result["people"].items():
        out.append(f"- {n}: {_m(p['total'])}" + ("  [manual adjustment]" if p.get("adjustment") else ""))
    if result["unassigned"]:
        u = result["unassigned"]
        out.append(f"- Not assigned yet: {', '.join(u['items'])} ({_m(u['total'])} including its share of charges)")
    out += ["", "Taxes, service charge and discounts are split in proportion to each person's share of the "
            "pre-tax bill. Amounts are rounded to the paisa so everyone's total adds up to the bill exactly."]
    for o in result["overrides"]:
        if o["kind"] == "total":
            out.append(f"Manual adjustment: {o['person']} set to {_m(o['new'])} (was {_m(o['original'])})."
                       + (f" Reason: {o['reason']}." if o["reason"] else ""))
        else:
            out.append(f"Manual adjustment: assignment of item {o['item_id']} was changed by hand.")
    out += [f"Note: {w}" for w in result["warnings"]]
    return "\n".join(out)
