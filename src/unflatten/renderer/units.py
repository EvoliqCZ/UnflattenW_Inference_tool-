"""Deterministic unit conversion (RFC002 OOXML mapping principles).

All AST length/ratio/angle values are integers in `mpt` (milli-points),
`ppm` (parts-per-million) or `mdeg` (milli-degrees). OOXML units are
derived via these centralized, exhaustively-tested pure functions —
never inline arithmetic scattered across the compiler.

Conversion table:
    1 twip        = 50 mpt
    1 half-point  = 500 mpt
    1 eighth-point = 125 mpt
    1 mpt         = 12.7 EMU   (emu = mpt * 127 // 10, rounded)
    word_pct      = ppm / 200
    ooxml_angle   = mdeg * 60  (exact, no rounding needed)

All rounding uses `round_half_away_from_zero` per RFC002.
"""

from __future__ import annotations


def round_half_away_from_zero(numerator: int, denominator: int) -> int:
    """Divide `numerator` by `denominator`, rounding half away from zero.

    E.g. round_half_away_from_zero(5, 2) == 3, round_half_away_from_zero(-5, 2) == -3.
    """
    if denominator == 0:
        raise ValueError("round_half_away_from_zero: denominator must not be zero")
    negative = (numerator < 0) != (denominator < 0)
    n, d = abs(numerator), abs(denominator)
    quotient, remainder = divmod(n, d)
    if 2 * remainder >= d:
        quotient += 1
    return -quotient if negative else quotient


def mpt_to_twip(mpt: int) -> int:
    """Convert milli-points to twips (1 twip = 50 mpt)."""
    return round_half_away_from_zero(mpt, 50)


def mpt_to_half_point(mpt: int) -> int:
    """Convert milli-points to half-points (1 half-point = 500 mpt)."""
    return round_half_away_from_zero(mpt, 500)


def mpt_to_eighth_point(mpt: int) -> int:
    """Convert milli-points to eighth-points (1/8 pt = 125 mpt)."""
    return round_half_away_from_zero(mpt, 125)


def mpt_to_emu(mpt: int) -> int:
    """Convert milli-points to EMU (1 mpt = 12.7 EMU == 127/10)."""
    return round_half_away_from_zero(mpt * 127, 10)


def ppm_to_word_pct(ppm: int) -> int:
    """Convert parts-per-million to Word's table percent unit (1/50 %)."""
    return round_half_away_from_zero(ppm, 200)


def mdeg_to_ooxml_angle(mdeg: int) -> int:
    """Convert milli-degrees to DrawingML 1/60000-degree units (exact)."""
    return mdeg * 60


def ppm_to_srcrect(ppm: int) -> int:
    """Convert parts-per-million to `a:srcRect` units (1/1000 of a percent).

    1_000_000 ppm == 100% == 100_000 srcRect units, so srcRect = ppm / 10.
    """
    return round_half_away_from_zero(ppm, 10)
