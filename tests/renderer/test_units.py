import pytest

from unflatten.renderer.units import (
    mdeg_to_ooxml_angle,
    mpt_to_eighth_point,
    mpt_to_emu,
    mpt_to_half_point,
    mpt_to_twip,
    ppm_to_word_pct,
    round_half_away_from_zero,
)


@pytest.mark.parametrize(
    "numerator,denominator,expected",
    [
        (5, 2, 3),
        (-5, 2, -3),
        (5, -2, -3),
        (-5, -2, 3),
        (4, 2, 2),
        (-4, 2, -2),
        (0, 5, 0),
        (1, 3, 0),
        (2, 3, 1),
        (-1, 3, 0),
        (-2, 3, -1),
    ],
)
def test_round_half_away_from_zero(numerator, denominator, expected):
    assert round_half_away_from_zero(numerator, denominator) == expected


def test_round_half_away_from_zero_zero_denominator_raises():
    with pytest.raises(ValueError):
        round_half_away_from_zero(1, 0)


@pytest.mark.parametrize(
    "mpt,expected",
    [(0, 0), (50, 1), (100, 2), (25, 1), (24, 0), (-50, -1), (-25, -1), (-24, 0)],
)
def test_mpt_to_twip(mpt, expected):
    assert mpt_to_twip(mpt) == expected


@pytest.mark.parametrize(
    "mpt,expected",
    [(0, 0), (500, 1), (11000, 22), (250, 1), (249, 0), (-500, -1)],
)
def test_mpt_to_half_point(mpt, expected):
    assert mpt_to_half_point(mpt) == expected


@pytest.mark.parametrize(
    "mpt,expected",
    [(0, 0), (125, 1), (250, 2), (62, 0), (63, 1), (-125, -1)],
)
def test_mpt_to_eighth_point(mpt, expected):
    assert mpt_to_eighth_point(mpt) == expected


@pytest.mark.parametrize(
    "mpt,expected",
    [(0, 0), (10, 127), (1000, 12700), (-10, -127)],
)
def test_mpt_to_emu(mpt, expected):
    assert mpt_to_emu(mpt) == expected


@pytest.mark.parametrize(
    "ppm,expected",
    [(0, 0), (200, 1), (1_000_000, 5000), (100, 1), (99, 0), (-200, -1)],
)
def test_ppm_to_word_pct(ppm, expected):
    assert ppm_to_word_pct(ppm) == expected


@pytest.mark.parametrize(
    "mdeg,expected",
    [(0, 0), (1000, 60000), (-1000, -60000), (1, 60)],
)
def test_mdeg_to_ooxml_angle(mdeg, expected):
    assert mdeg_to_ooxml_angle(mdeg) == expected
