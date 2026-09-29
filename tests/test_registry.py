"""Tests for the Champion vs Challenger promotion rule (src/register_model.py)."""

from src.register_model import should_promote


def test_first_model_is_always_promoted():
    assert should_promote(0.70, None) is True


def test_better_challenger_is_promoted():
    assert should_promote(0.78, 0.77) is True


def test_worse_challenger_is_not_promoted():
    assert should_promote(0.74, 0.77) is False


def test_equal_challenger_is_not_promoted():
    # Same score (e.g. retraining on the same data with the same seed): keep the champion.
    assert should_promote(0.775, 0.775) is False


def test_minimum_improvement_is_respected():
    assert should_promote(0.778, 0.775, min_improvement=0.005) is False
    assert should_promote(0.781, 0.775, min_improvement=0.005) is True
