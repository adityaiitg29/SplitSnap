"""Money helpers. Inside the engine every amount is an integer number of paise,
so nothing is ever lost to floating point and totals always add up exactly."""
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from fractions import Fraction


def to_paise(x):
    """'1,234.50', '₹40', 18.5, Decimal('7') -> integer paise. Raises ValueError if unusable."""
    if x is None or isinstance(x, bool):
        raise ValueError(f"not a money amount: {x!r}")
    if isinstance(x, str):
        x = re.sub(r"₹|rs\.?|inr|,|\s", "", x, flags=re.I)
    try:
        d = Decimal(str(x))
    except InvalidOperation:
        raise ValueError(f"not a money amount: {x!r}")
    if not d.is_finite():
        raise ValueError(f"not a money amount: {x!r}")
    return int((d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) * 100).to_integral_value())


def rupees(p):
    """Integer paise -> float rupees (2 decimals), for JSON and display."""
    return round(p / 100, 2)


def _group_indian(n):
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail, parts = s[:-3], s[-3:], []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def fmt_inr(p, always_decimals=False):
    """Paise -> '₹1,23,456.50'. Whole rupees print without decimals unless asked."""
    neg, p = p < 0, abs(p)
    r, ps = divmod(p, 100)
    body = _group_indian(r) + (f".{ps:02d}" if ps or always_decimals else "")
    return ("-" if neg else "") + "₹" + body


def allocate(amount, weights):
    """Split an integer amount of paise across people in proportion to `weights`
    (name -> non-negative Fraction/int) using the largest-remainder method.

    The parts always sum to `amount` exactly. Leftover paise go to whoever has the
    largest fractional remainder; ties go to the larger weight, then to the order of
    `weights` (which the engine keeps equal to the party order)."""
    names = list(weights)
    total_w = sum(Fraction(weights[n]) for n in names)
    if total_w <= 0:
        raise ValueError("weights must have a positive sum")
    sign, a = (-1 if amount < 0 else 1), abs(amount)
    exact = {n: Fraction(a) * Fraction(weights[n]) / total_w for n in names}
    base = {n: exact[n].numerator // exact[n].denominator for n in names}
    leftover = a - sum(base.values())
    ranked = sorted(names, key=lambda n: (-(exact[n] - base[n]), -Fraction(weights[n]), names.index(n)))
    for n in ranked[:leftover]:
        base[n] += 1
    return {n: sign * base[n] for n in names}
