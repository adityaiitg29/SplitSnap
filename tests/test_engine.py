import copy
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splitsnap_engine import *          # noqa: E402
from splitsnap_engine.money import _group_indian   # noqa: E402

PARTY = ["Riya", "Aman", "Kabir"]


def ps_bill():
    raw = {"items": [{"name": "Paneer Butter Masala", "qty": 1, "unit_price": 220, "total": 220},
                     {"name": "Garlic Naan", "qty": 3, "unit_price": 40, "total": 120},
                     {"name": "Veg Biryani", "qty": 1, "unit_price": 260, "total": 260},
                     {"name": "Cold Coffee", "qty": 1, "unit_price": 140, "total": 140}],
           "charges": [{"type": "CGST", "amount": 18.5}, {"type": "SGST", "amount": 18.5},
                       {"type": "Service Charge", "amount": 40}, {"type": "Discount", "amount": -50}],
           "subtotal": 740, "grand_total": 767}
    return normalize_bill(raw)[0]


ASSIGN = {"i0": "Riya", "i1": {"Riya": 1, "Aman": 1, "Kabir": 1}, "i2": "Aman", "i3": "Kabir"}


def totals(res):
    return {n: p["total"] for n, p in res["people"].items()}


class Money(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(to_paise("₹1,234.50"), 123450)
        self.assertEqual(to_paise("Rs. 40"), 4000)
        self.assertEqual(to_paise(18.5), 1850)
        self.assertEqual(to_paise(-50), -5000)
        for bad in (None, "abc", float("nan"), True):
            with self.assertRaises(ValueError):
                to_paise(bad)

    def test_indian_grouping(self):
        self.assertEqual(_group_indian(123456), "1,23,456")
        self.assertEqual(fmt_inr(12345678), "₹1,23,456.78")
        self.assertEqual(fmt_inr(-5000), "-₹50")
        self.assertEqual(fmt_inr(3350), "₹33.50")

    def test_allocate_always_sums_exactly(self):
        rng = random.Random(7)
        for _ in range(3000):
            amount = rng.randint(-100000, 100000)
            weights = {f"p{i}": rng.randint(0, 500) for i in range(rng.randint(1, 8))}
            if sum(weights.values()) == 0:
                continue
            self.assertEqual(sum(allocate(amount, weights).values()), amount)

    def test_allocate_is_fair_and_never_pays_zero_weight(self):
        self.assertEqual(sorted(allocate(100, {"a": 1, "b": 1, "c": 1}).values()), [33, 33, 34])
        self.assertEqual(allocate(10, {"a": 1, "b": 0})["b"], 0)


class PSSampleBill(unittest.TestCase):
    """Numbers worked out by hand: each charge split by pre-tax share, largest remainder."""

    def test_exact_totals(self):
        res = split_bill(ps_bill(), PARTY, ASSIGN)
        self.assertEqual(totals(res), {"Riya": 269.48, "Aman": 310.95, "Kabir": 186.57})
        self.assertAlmostEqual(sum(totals(res).values()), 767.0, places=2)
        self.assertTrue(res["reconciled"])
        self.assertEqual(res["warnings"], [])

    def test_charges_shown_separately(self):
        s = split_bill(ps_bill(), PARTY, ASSIGN)["summary"]
        self.assertEqual((s["items"], s["taxes"], s["service"], s["discount"], s["total"]), (740, 37, 40, -50, 767))
        self.assertEqual(s["line"], "Items: ₹740, Taxes: ₹37, Service Charge: ₹40, Discount: -₹50, Total: ₹767")

    def test_not_a_headcount_split(self):
        res = split_bill(ps_bill(), PARTY, ASSIGN)
        self.assertNotEqual(res["people"]["Riya"]["total"], round(767 / 3, 2))


class SharingAndParty(unittest.TestCase):
    def test_fraction_shares(self):
        bill = normalize_bill({"items": [{"name": "Platter", "qty": 1, "total": 300}], "charges": []})[0]
        res = split_bill(bill, ["A", "B"], {"i0": {"A": "1/3", "B": "2/3"}})
        self.assertEqual(totals(res), {"A": 100, "B": 200})

    def test_person_with_nothing_owes_nothing(self):
        res = split_bill(ps_bill(), PARTY + ["Zoya"], ASSIGN)
        self.assertEqual(res["people"]["Zoya"]["total"], 0)
        self.assertEqual(sum(totals(res).values()), 767)

    def test_all_and_list_shorthand(self):
        res = split_bill(ps_bill(), PARTY, {"i0": "all", "i1": ["Riya", "Aman"], "i2": "Aman", "i3": ["Kabir"]})
        self.assertEqual(round(sum(totals(res).values()), 2), 767)

    def test_repeated_name_counts_twice(self):
        bill = normalize_bill({"items": [{"name": "Naan", "qty": 3, "total": 90}], "charges": []})[0]
        res = split_bill(bill, ["A", "B"], {"i0": ["A", "A", "B"]})
        self.assertEqual(totals(res), {"A": 60, "B": 30})

    def test_positional_keys_and_case_insensitive_names(self):
        res = split_bill(ps_bill(), PARTY, {0: "riya", 1: "all", 2: "AMAN", 3: "kabir"})
        self.assertEqual(totals(res), totals(split_bill(ps_bill(), PARTY, ASSIGN)))

    def test_unknown_member_and_bad_share_raise(self):
        with self.assertRaises(UnknownMemberError):
            split_bill(ps_bill(), PARTY, {"i0": "Nobody"})
        with self.assertRaises(InvalidShareError):
            split_bill(ps_bill(), PARTY, {"i0": {"Riya": "-1"}})
        with self.assertRaises(InvalidShareError):
            split_bill(ps_bill(), PARTY, {"i0": {"Riya": "lots"}})
        with self.assertRaises(SplitError):
            split_bill(ps_bill(), ["A", "a"], {})

    def test_remove_member_mid_flow(self):
        bill = ps_bill()
        party, assign, orphaned = remove_member(ASSIGN, PARTY, "Kabir", bill["items"])
        self.assertEqual(party, ["Riya", "Aman"])
        self.assertEqual(orphaned, ["i3"])            # Cold Coffee was only Kabir's
        res = split_bill(bill, party, assign)
        self.assertEqual(res["unassigned"]["items"], ["Cold Coffee"])
        self.assertFalse(res["reconciled"])
        self.assertEqual(round(sum(totals(res).values()) + res["unassigned"]["total"], 2), 767)

    def test_add_and_rename_member(self):
        self.assertEqual(add_member(PARTY, "Zoya"), PARTY + ["Zoya"])
        with self.assertRaises(SplitError):
            add_member(PARTY, "riya")
        party, assign = rename_member(ASSIGN, PARTY, "Riya", "Riyaa", ps_bill()["items"])
        res = split_bill(ps_bill(), party, assign)
        self.assertIn("Riyaa", res["people"])
        self.assertEqual(res["people"]["Riyaa"]["total"], 269.48)


class Unassigned(unittest.TestCase):
    def test_report_keeps_everything_reconciled_but_flags_it(self):
        res = split_bill(ps_bill(), PARTY, {"i0": "Riya", "i2": "Aman"})
        self.assertFalse(res["reconciled"])
        self.assertEqual(sorted(res["unassigned"]["items"]), ["Cold Coffee", "Garlic Naan"])
        # assigned people don't carry tax for food nobody claimed
        everyone = sum(totals(res).values()) + res["unassigned"]["total"]
        self.assertAlmostEqual(everyone, 767, places=2)
        self.assertTrue(any("not assigned" in w for w in res["warnings"]))

    def test_equal_all_and_error_policies(self):
        res = split_bill(ps_bill(), PARTY, {"i0": "Riya"}, on_unassigned="equal_all")
        self.assertTrue(res["reconciled"])
        self.assertAlmostEqual(sum(totals(res).values()), 767, places=2)
        with self.assertRaises(UnassignedItemsError):
            split_bill(ps_bill(), PARTY, {"i0": "Riya"}, on_unassigned="error")


class ChargesAndReconcile(unittest.TestCase):
    def test_equal_basis_charge(self):
        raw = {"items": [{"name": "A", "qty": 1, "total": 300}, {"name": "B", "qty": 1, "total": 100}],
               "charges": [{"type": "Cover Charge", "amount": 100, "basis": "equal"}]}
        res = split_bill(normalize_bill(raw)[0], ["X", "Y"], {"i0": "X", "i1": "Y"})
        self.assertEqual(totals(res), {"X": 350, "Y": 150})

    def test_discount_sign_is_repaired(self):
        bill, warnings, _ = normalize_bill({"items": [{"name": "A", "qty": 1, "total": 200}],
                                            "charges": [{"type": "Discount", "amount": 50}]})
        self.assertEqual(bill["charges"][0]["amount"], -50)
        self.assertTrue(any("deduction" in w for w in warnings))

    def test_small_gap_becomes_inferred_round_off(self):
        raw = {"items": [{"name": "A", "qty": 1, "total": 99.6}], "charges": [], "grand_total": 100}
        res = split_bill(normalize_bill(raw)[0], ["X"], {"i0": "X"})
        self.assertEqual(res["people"]["X"]["total"], 100)
        self.assertEqual(res["bill"]["charges"][-1]["type"], "Round Off (inferred)")

    def test_big_gap_is_warned_not_silently_fixed(self):
        raw = {"items": [{"name": "A", "qty": 1, "total": 100}], "charges": [], "grand_total": 180}
        res = split_bill(normalize_bill(raw)[0], ["X"], {"i0": "X"})
        self.assertEqual(res["people"]["X"]["total"], 100)
        self.assertTrue(any("differs" in w for w in res["warnings"]))
        forced = split_bill(normalize_bill(raw)[0], ["X"], {"i0": "X"}, reconcile="force")
        self.assertEqual(forced["people"]["X"]["total"], 180)

    def test_all_free_items_still_split_charges_equally(self):
        raw = {"items": [{"name": "Free tea", "qty": 2, "total": 0}], "charges": [{"type": "Service Charge", "amount": 10}]}
        res = split_bill(normalize_bill(raw)[0], ["X", "Y"], {"i0": "all"})
        self.assertEqual(totals(res), {"X": 5, "Y": 5})

    def test_rounding_never_loses_a_paisa(self):
        raw = {"items": [{"name": "A", "qty": 1, "total": 100}], "charges": [{"type": "GST", "amount": 5}]}
        res = split_bill(normalize_bill(raw)[0], ["A1", "B1", "C1"], {"i0": "all"})
        self.assertAlmostEqual(sum(totals(res).values()), 105, places=2)


class Overrides(unittest.TestCase):
    def test_override_redistributes_and_flags(self):
        base = split_bill(ps_bill(), PARTY, ASSIGN)
        res = apply_total_override(base, "Riya", 280, reason="extra drink")
        self.assertEqual(res["people"]["Riya"]["total"], 280)
        self.assertAlmostEqual(sum(totals(res).values()), 767, places=2)
        self.assertTrue(res["manual"] and res["reconciled"])
        self.assertEqual(res["people"]["Riya"]["adjustment"]["type"], "override")
        self.assertEqual(res["people"]["Aman"]["adjustment"]["caused_by"], "Riya")
        self.assertEqual(totals(base)["Riya"], 269.48)          # original untouched

    def test_override_without_redistribution_is_marked_unreconciled(self):
        res = apply_total_override(split_bill(ps_bill(), PARTY, ASSIGN), "Riya", 300, redistribute="none")
        self.assertFalse(res["reconciled"])
        self.assertTrue(any("add up to" in w for w in res["warnings"]))

    def test_second_override_pins_first_and_keeps_original(self):
        r1 = apply_total_override(split_bill(ps_bill(), PARTY, ASSIGN), "Riya", 280)
        r2 = apply_total_override(r1, "Aman", 300)
        self.assertEqual(r2["people"]["Riya"]["total"], 280)    # pinned
        self.assertAlmostEqual(sum(totals(r2).values()), 767, places=2)
        self.assertEqual(r2["people"]["Aman"]["adjustment"]["original_total"], 310.95)

    def test_override_errors(self):
        base = split_bill(ps_bill(), PARTY, ASSIGN)
        with self.assertRaises(UnknownMemberError):
            apply_total_override(base, "Nobody", 10)
        with self.assertRaises(SplitError):
            apply_total_override(base, "Riya", -1)
        with self.assertRaises(SplitError):
            apply_total_override(base, "Riya", 800)             # more than the whole bill: others would go below zero

    def test_manual_assignment_flag(self):
        res = split_bill(ps_bill(), PARTY, ASSIGN, manual_items=["i3"])
        self.assertTrue(res["manual"])
        self.assertTrue(res["people"]["Kabir"]["lines"][-1]["manual"])

    def test_settle_up(self):
        res = split_bill(ps_bill(), PARTY, ASSIGN)
        transfers = settle_up(res, "Aman")
        self.assertEqual({(t["from"], t["to"]) for t in transfers}, {("Riya", "Aman"), ("Kabir", "Aman")})
        self.assertAlmostEqual(sum(t["amount"] for t in transfers), 269.48 + 186.57, places=2)
        split_pay = settle_up(res, {"Aman": 500, "Kabir": 267})
        self.assertAlmostEqual(sum(t["amount"] for t in split_pay), 269.48 + 0, places=1)


class NormalizeFromModel(unittest.TestCase):
    def test_messy_model_output(self):
        pred = {"items": [{"name": "Dal", "qty": 2, "unit_price": 100, "total": 250},     # inconsistent
                          {"name": "", "qty": None, "unit_price": None, "total": 40},
                          {"name": "Ghost", "qty": 1, "unit_price": None, "total": None},   # unreadable
                          {"name": "Tea", "qty": 0, "unit_price": 20, "total": None}],
                "charges": [{"type": "CGST", "amount": None}, {"type": "SGST", "amount": "2.5"}],
                "subtotal": 500, "grand_total": None}
        bill, warnings, _ = normalize_bill(pred)
        self.assertEqual([i["name"] for i in bill["items"]], ["Dal", "Item 2", "Tea"])
        self.assertEqual(bill["items"][2]["total"], 20)                                     # qty 0 repaired to 1
        self.assertEqual(len(bill["charges"]), 1)
        self.assertTrue(any("Ghost" in w for w in warnings))
        self.assertTrue(any("subtotal" in w for w in warnings))
        self.assertTrue(any("line total" in w for w in warnings))

    def test_confidence_flags_map_to_ids(self):
        pred = {"items": [{"name": "A", "qty": 1, "unit_price": 10, "total": 10},
                          {"name": "B", "qty": 1, "unit_price": 20, "total": 20}],
                "charges": [{"type": "CGST", "amount": 5}], "subtotal": 30, "grand_total": 35}
        conf = {"items[1].total": {"min": 0.41, "mean": 0.7}, "items[0].name": {"min": 0.99, "mean": 0.99},
                "charges[0].amount": {"min": 0.6, "mean": 0.8}, "grand_total": {"min": 0.3, "mean": 0.5},
                "items[9].total": {"min": 0.1, "mean": 0.1}}
        _, _, flags = normalize_bill(pred, conf)
        self.assertEqual([(f["target"], f["id"], f["field"]) for f in flags],
                         [("bill", None, "grand_total"), ("item", "i1", "total"), ("charge", "c0", "amount")])

    def test_charge_categories(self):
        cat = charge_category
        self.assertEqual([cat("CGST"), cat("Service Charge"), cat("Discount 10%"), cat("Round Off"),
                          cat("Packaging"), cat("VAT 5%"), cat("Loyalty", -10)],
                         ["tax", "service", "discount", "rounding", "other", "tax", "discount"])

    def test_edits_after_correction_propagate(self):
        """R3: fix a wrong price in the corrected bill and the split follows."""
        bill = ps_bill()
        bill["items"][2]["total"] = bill["items"][2]["unit_price"] = 300
        res = split_bill(bill, PARTY, ASSIGN, reconcile="strict")
        self.assertEqual(res["people"]["Aman"]["items_subtotal"], 340)
        self.assertTrue(any("differs" in w for w in res["warnings"]))


class Explanations(unittest.TestCase):
    def setUp(self):
        self.res = split_bill(ps_bill(), PARTY, ASSIGN)

    def test_every_number_in_the_text_matches_the_table(self):
        text = explain_person(self.res, "Riya")
        for s in ("₹220", "₹40", "₹260", "35.1%", "₹6.50", "₹14.05", "-₹17.57", "₹9.48", "₹269.48"):
            self.assertIn(s, text)

    def test_names_flow_through(self):
        for n, t in explain_all(self.res).items():
            self.assertTrue(t.startswith(n))

    def test_short_style_and_bill_summary(self):
        self.assertIn("Riya owes ₹269.48", explain_person(self.res, "Riya", "short"))
        self.assertIn("Total: ₹767", explain_bill(self.res))

    def test_manual_adjustments_are_flagged(self):
        res = apply_total_override(self.res, "Riya", 280, reason="extra drink")
        self.assertIn("MANUAL ADJUSTMENT", explain_person(res, "Riya"))
        self.assertIn("extra drink", explain_person(res, "Riya"))
        self.assertIn("MANUAL ADJUSTMENT", explain_person(res, "Aman"))
        self.assertIn("[manual adjustment]", explain_bill(res))

    def test_equal_charge_and_unclaimed_item_wording(self):
        raw = {"items": [{"name": "A", "qty": 1, "total": 300}], "charges": [{"type": "Cover Charge", "amount": 100, "basis": "equal"}]}
        res = split_bill(normalize_bill(raw)[0], ["X", "Y"], {}, on_unassigned="equal_all")
        text = explain_person(res, "X")
        self.assertIn("split equally among 2 people", text)
        self.assertIn("nobody claimed it", text)


class Properties(unittest.TestCase):
    def test_random_bills_always_add_up(self):
        rng = random.Random(11)
        for _ in range(500):
            party = [f"P{i}" for i in range(rng.randint(1, 7))]
            items = [{"name": f"it{i}", "qty": rng.randint(1, 4), "total": rng.randint(1, 90000) / 100}
                     for i in range(rng.randint(1, 10))]
            charges = [{"type": t, "amount": rng.randint(-5000, 9000) / 100}
                       for t in rng.sample(["CGST", "SGST", "Service Charge", "Discount", "Round Off", "Packaging"], rng.randint(0, 5))]
            bill = normalize_bill({"items": items, "charges": charges})[0]
            assign = {}
            for it in bill["items"]:
                k = rng.randint(0, len(party))
                if k:
                    assign[it["id"]] = {p: rng.randint(1, 4) for p in rng.sample(party, k)}
            res = split_bill(bill, party, assign)
            left = to_paise(res["unassigned"]["total"]) if res["unassigned"] else 0
            computed = sum(to_paise(i["total"]) for i in bill["items"]) + sum(to_paise(c["amount"]) for c in bill["charges"])
            self.assertEqual(sum(to_paise(p["total"]) for p in res["people"].values()) + left, computed)
            for n in party:
                explain_person(res, n)                       # must never crash
            explain_bill(res)


if __name__ == "__main__":
    unittest.main(verbosity=1)
