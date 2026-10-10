"""The split itself.

Rules (these are what the explanations quote, so they are kept simple and consistent):
  1. Each item's price is divided among the people who shared it, by weight
     (equal weights = equal shares; {"Riya": 2, "Aman": 1} = two units and one unit; "1/3" works too).
  2. Every non-item charge (CGST, SGST, service charge, discount, round-off) is divided in
     proportion to each person's share of the PRE-TAX bill. Charges with basis "equal"
     (cover charge, delivery fee) are divided equally per head instead.
  3. All arithmetic is in integer paise with largest-remainder rounding, so the people's
     totals add up to the bill to the paisa, always.
"""
import copy
from collections import OrderedDict
from fractions import Fraction

from .bill import normalize_bill, summarize_charges
from .money import allocate, fmt_inr, rupees, to_paise

UNASSIGNED = "(unassigned)"


class SplitError(ValueError):
    pass


class UnknownMemberError(SplitError):
    pass


class InvalidShareError(SplitError):
    pass


class UnassignedItemsError(SplitError):
    pass


# ----------------------------------------------------------------- party and shares
def _clean_party(party):
    names = [str(p).strip() for p in party]
    if not names or any(not n for n in names):
        raise SplitError("The party needs at least one named member.")
    if len({n.casefold() for n in names}) != len(names):
        raise SplitError("Two members have the same name.")
    return names


def _weight(w, who):
    try:
        f = w if isinstance(w, Fraction) else Fraction(str(w).strip())
    except (ValueError, ZeroDivisionError):
        raise InvalidShareError(f"Can't read the share '{w}' for {who}.")
    if f < 0:
        raise InvalidShareError(f"Share for {who} can't be negative.")
    return f


def parse_shares(spec, party, label=""):
    """'all' | 'Riya' | ['Riya','Aman'] | {'Riya': 2, 'Aman': '1/3'} -> OrderedDict name -> Fraction.
    Empty result means nobody has claimed the item. Unknown names raise UnknownMemberError."""
    canon = {p.casefold(): p for p in party}
    got = {}

    def add(name, w):
        key = canon.get(str(name).strip().casefold())
        if key is None:
            raise UnknownMemberError(f"'{name}' is not in the party (item: {label}).")
        got[key] = got.get(key, Fraction(0)) + w

    if spec is None:
        pass
    elif isinstance(spec, str):
        if spec.strip().casefold() in ("all", "everyone", "*"):
            for p in party:
                add(p, Fraction(1))
        else:
            add(spec, Fraction(1))
    elif isinstance(spec, dict):
        for n, w in spec.items():
            add(n, _weight(w, n))
    else:                                   # list/tuple/set: a name repeated twice counts twice
        for n in spec:
            add(n, Fraction(1))
    return OrderedDict((p, got[p]) for p in party if got.get(p, 0) > 0)


def _resolve_keys(assignments, items):
    """Assignments may be keyed by item id ('i2') or by position (2)."""
    ids, out = [it["id"] for it in items], {}
    for k, v in (assignments or {}).items():
        if k in ids:
            out[k] = v
        elif isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
            if not 0 <= int(k) < len(ids):
                raise SplitError(f"There is no item number {k}.")
            out[ids[int(k)]] = v
        else:
            raise SplitError(f"There is no item '{k}'.")
    return out


def normalize_assignments(assignments, party, items):
    """Canonical, JSON-friendly form: {item_id: {name: 'weight'}}. Good app state to store."""
    party = _clean_party(party)
    return {iid: {n: str(w) for n, w in parse_shares(spec, party, iid).items()}
            for iid, spec in _resolve_keys(assignments, items).items()}


def add_member(party, name):
    return _clean_party(list(party) + [name])


def remove_member(assignments, party, name, items):
    """Drop someone mid-flow. Their items go to whoever else shared them; items nobody else
    shared become unassigned. Returns (new_party, new_assignments, orphaned_item_ids)."""
    party = _clean_party(party)
    canon = {p.casefold(): p for p in party}
    if name.strip().casefold() not in canon:
        raise UnknownMemberError(f"'{name}' is not in the party.")
    gone = canon[name.strip().casefold()]
    new_party = [p for p in party if p != gone]
    new, orphaned = {}, []
    for iid, shares in normalize_assignments(assignments, party, items).items():
        rest = {n: w for n, w in shares.items() if n != gone}
        if rest:
            new[iid] = rest
        else:
            orphaned.append(iid)
    return new_party, new, orphaned


def rename_member(assignments, party, old, new_name, items):
    party = _clean_party(party)
    canon = {p.casefold(): p for p in party}
    if old.strip().casefold() not in canon:
        raise UnknownMemberError(f"'{old}' is not in the party.")
    old = canon[old.strip().casefold()]
    new_party = _clean_party([new_name.strip() if p == old else p for p in party])
    new = {iid: {(new_name.strip() if n == old else n): w for n, w in shares.items()}
           for iid, shares in normalize_assignments(assignments, party, items).items()}
    return new_party, new


# ----------------------------------------------------------------------- the split
def split_bill(bill, party, assignments, *, on_unassigned="report", reconcile="auto",
               tolerance=1.0, manual_items=()):
    """Split `bill` among `party`.

    assignments    {item_id (or position): who shared it}, see parse_shares
    on_unassigned  'report'    leave unclaimed items in an '(unassigned)' bucket and warn (default)
                   'equal_all' split them equally among everyone
                   'error'     raise UnassignedItemsError
    reconcile      what to do when printed total != items + charges
                   'auto'   add a small inferred Round Off if the gap is within `tolerance` rupees,
                            otherwise warn only (a bigger gap means a misread line, a human should look)
                   'strict' never adjust; 'force' always add the gap as an inferred charge
    manual_items   item ids whose assignment the user changed by hand after the first split
    """
    warnings = []
    if not bill.get("_normalized"):
        bill, w, _ = normalize_bill(bill)
        warnings += w
    party = _clean_party(party)
    items = bill["items"]
    amap = _resolve_keys(assignments, items)

    # 1. items -> people
    lines = OrderedDict((p, []) for p in party)
    sub = OrderedDict((p, 0) for p in party)
    sub[UNASSIGNED] = 0
    unassigned_items = []
    for it in items:
        tot = to_paise(it["total"])
        w = parse_shares(amap.get(it["id"]), party, it["name"])
        defaulted = False
        if not w:
            if on_unassigned == "error":
                raise UnassignedItemsError(f"'{it['name']}' is not assigned to anyone.")
            if on_unassigned == "equal_all":
                w, defaulted = OrderedDict((p, Fraction(1)) for p in party), True
            else:
                unassigned_items.append(it)
                sub[UNASSIGNED] += tot
                warnings.append(f"'{it['name']}' is not assigned to anyone yet.")
                continue
        total_w = sum(w.values())
        qty = it["qty"]
        units_mode = (qty > 1 and float(qty).is_integer() and total_w == int(qty)
                      and all(x.denominator == 1 for x in w.values()))
        for p, amt in allocate(tot, w).items():
            share = w[p] / total_w
            lines[p].append({
                "item_id": it["id"], "name": it["name"], "qty": qty, "item_total": it["total"],
                "share": str(share), "share_fraction": round(float(share), 6),
                "units": int(w[p]) if units_mode else None,
                "amount": rupees(amt), "shared_with": [q for q in w if q != p],
                "defaulted": defaulted, "manual": it["id"] in set(manual_items)})
            sub[p] += amt

    # 2. reconcile with the printed total
    charges = [dict(c) for c in bill["charges"]]
    items_p = sum(to_paise(i["total"]) for i in items)
    computed = items_p + sum(to_paise(c["amount"]) for c in charges)
    stated = to_paise(bill["stated_total"]) if bill.get("stated_total") is not None else None
    if stated is not None and stated != computed:
        gap = stated - computed
        small = abs(gap) <= to_paise(tolerance)
        if reconcile == "force" or (reconcile == "auto" and small):
            charges.append({"id": "inferred", "type": "Round Off (inferred)" if small else "Other charges (inferred)",
                            "category": "rounding" if small else "other", "amount": rupees(gap),
                            "basis": "proportional", "inferred": True})
            computed += gap
            warnings.append(f"Printed total is {fmt_inr(stated)}; added {fmt_inr(gap)} as "
                            f"{'a round-off' if small else 'an unlisted charge'} so the split matches it.")
        else:
            warnings.append(f"Printed total {fmt_inr(stated)} differs from items + charges {fmt_inr(computed)} "
                            f"by {fmt_inr(abs(gap))}. Check the items, or add the missing charge.")

    # 3. charges -> people
    per_charge = []
    for c in charges:
        amt = to_paise(c["amount"])
        if c.get("basis") == "equal":
            weights = OrderedDict((p, Fraction(1)) for p in party)
        else:
            weights = OrderedDict((k, Fraction(v)) for k, v in sub.items() if v > 0)
            if not weights or any(v < 0 for v in sub.values()):
                weights = OrderedDict((p, Fraction(1)) for p in party)
                warnings.append(f"'{c['type']}' was split equally: item subtotals can't be used as weights.")
        per_charge.append(allocate(amt, weights))

    people = OrderedDict()
    for p in party:
        mine = [{"id": c["id"], "type": c["type"], "category": c["category"], "basis": c.get("basis", "proportional"),
                 "amount": rupees(a.get(p, 0)), "bill_amount": c["amount"]}
                for c, a in zip(charges, per_charge)]
        charges_p = sum(to_paise(m["amount"]) for m in mine)
        people[p] = {"name": p, "lines": lines[p], "items_subtotal": rupees(sub[p]),
                     "pretax_fraction": round(sub[p] / items_p, 6) if items_p else 0.0,
                     "charges": mine, "charges_total": rupees(charges_p),
                     "total": rupees(sub[p] + charges_p), "adjustment": None}

    unassigned = None
    if unassigned_items:
        ch = sum(a.get(UNASSIGNED, 0) for a in per_charge)
        unassigned = {"items": [i["name"] for i in unassigned_items], "item_ids": [i["id"] for i in unassigned_items],
                      "items_subtotal": rupees(sub[UNASSIGNED]), "charges_total": rupees(ch),
                      "total": rupees(sub[UNASSIGNED] + ch)}

    paid_out = sum(to_paise(p["total"]) for p in people.values())
    if paid_out + (to_paise(unassigned["total"]) if unassigned else 0) != computed:
        raise AssertionError("internal error: totals do not add up")   # cannot happen; guards future edits

    summary = summarize_charges({"items": items, "charges": charges})
    overrides = [{"kind": "assignment", "item_id": i, "reason": "assignment changed by hand"} for i in manual_items]
    return {"currency": "INR", "party": party, "people": people, "unassigned": unassigned,
            "bill": {"items_total": rupees(items_p), "charges": charges,
                     "charges_total": rupees(computed - items_p), "computed_total": rupees(computed),
                     "stated_total": bill.get("stated_total")},
            "summary": summary, "warnings": warnings, "overrides": overrides, "manual": bool(manual_items),
            "reconciled": unassigned is None}


# ------------------------------------------------------------------------ overrides
def apply_total_override(result, name, new_total, reason="", redistribute="others"):
    """Set one person's final amount by hand (R7). Returns a new result; the original is untouched.

    redistribute  'others' (default) spreads the difference over everyone else in proportion to what
                  they owe, skipping anyone already overridden, so the bill still adds up.
                  'none' changes only this person; the result is then marked not reconciled.
    Everything touched is flagged under person['adjustment'] and result['overrides']."""
    res = copy.deepcopy(result)
    if name not in res["people"]:
        raise UnknownMemberError(f"'{name}' is not in the party.")
    new_p = to_paise(new_total)
    if new_p < 0:
        raise SplitError("An amount can't be negative.")
    person = res["people"][name]
    old_p = to_paise(person["total"])
    delta = new_p - old_p
    if delta == 0:
        return res

    def first_original(p, current):
        return (p.get("adjustment") or {}).get("original_total", rupees(current))

    person["adjustment"] = {"manual": True, "type": "override", "original_total": first_original(person, old_p),
                            "delta": rupees(delta), "reason": reason}
    person["total"] = rupees(new_p)
    entry = {"kind": "total", "person": name, "original": rupees(old_p), "new": rupees(new_p),
             "delta": rupees(delta), "reason": reason, "redistribute": redistribute}

    free = [n for n, p in res["people"].items()
            if n != name and (p.get("adjustment") or {}).get("type") != "override"]
    if redistribute == "others" and free:
        weights = OrderedDict((n, Fraction(to_paise(res["people"][n]["total"]))) for n in free)
        if sum(weights.values()) == 0:
            weights = OrderedDict((n, Fraction(1)) for n in free)
        for n, d in allocate(-delta, weights).items():
            if d == 0:
                continue
            other = res["people"][n]
            before = to_paise(other["total"])
            if before + d < 0:
                raise SplitError(f"That would make {n}'s amount negative.")
            other["adjustment"] = {"manual": True, "type": "redistributed", "caused_by": name,
                                   "original_total": first_original(other, before),
                                   "delta": rupees(d), "reason": reason}
            other["total"] = rupees(before + d)
    else:
        entry["redistribute"] = "none"

    res["overrides"].append(entry)
    res["manual"] = True
    paid = sum(to_paise(p["total"]) for p in res["people"].values())
    left = to_paise(res["unassigned"]["total"]) if res["unassigned"] else 0
    res["reconciled"] = res["unassigned"] is None and paid + left == to_paise(res["bill"]["computed_total"])
    if paid + left != to_paise(res["bill"]["computed_total"]):
        res["warnings"].append(f"After the manual change, people's amounts add up to {fmt_inr(paid)} "
                               f"but the bill is {fmt_inr(to_paise(res['bill']['computed_total']))}.")
    return res


def settle_up(result, paid_by):
    """Who pays whom. paid_by is a name (that person paid the whole bill) or {name: amount_paid}.
    Returns the fewest simple transfers: [{'from', 'to', 'amount'}]."""
    owed = {n: to_paise(p["total"]) for n, p in result["people"].items()}
    if isinstance(paid_by, str):
        if paid_by not in owed:
            raise UnknownMemberError(f"'{paid_by}' is not in the party.")
        paid = {paid_by: sum(owed.values())}
    else:
        paid = {}
        for n, a in paid_by.items():
            if n not in owed:
                raise UnknownMemberError(f"'{n}' is not in the party.")
            paid[n] = to_paise(a)
    net = {n: paid.get(n, 0) - owed[n] for n in owed}
    creditors = sorted(([n, v] for n, v in net.items() if v > 0), key=lambda x: -x[1])
    debtors = sorted(([n, -v] for n, v in net.items() if v < 0), key=lambda x: -x[1])
    out, i, j = [], 0, 0
    while i < len(debtors) and j < len(creditors):
        pay = min(debtors[i][1], creditors[j][1])
        out.append({"from": debtors[i][0], "to": creditors[j][0], "amount": rupees(pay)})
        debtors[i][1] -= pay
        creditors[j][1] -= pay
        i += debtors[i][1] == 0
        j += creditors[j][1] == 0
    return out
