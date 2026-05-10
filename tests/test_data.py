"""Tests for data loaders and splits."""

from __future__ import annotations

import pandas as pd
import pytest

from promoternet.data import (
    _parse_name,
    leave_element_out_split,
    random_split,
)


def test_parse_name_neg_control_returns_none():
    assert _parse_name("neg_control_885001:885151") is None


def test_parse_name_simple_variant():
    name = "noUP_34G->T/30A->C_ECK125137108_12T->G/11A->T/9A->G/8A->T/7T->A_bg4323949:4324099"
    parsed = _parse_name(name)
    assert parsed is not None
    up, m35, spacer, m10, bg = parsed
    assert up == "noUP"
    assert m35 == "34G->T/30A->C"
    assert spacer == "ECK125137108"
    assert m10 == "12T->G/11A->T/9A->G/8A->T/7T->A"
    assert bg == "bg4323949:4324099"


def test_parse_name_compound_up_prefix():
    name = "gourse_326fold_up_35T->C/33G->T/31G->C_ECK125136938_9A->G/8A->T_bg463205:463355"
    parsed = _parse_name(name)
    assert parsed is not None
    up, m35, spacer, m10, bg = parsed
    assert up == "gourse_326fold_up"
    assert m35 == "35T->C/33G->T/31G->C"
    assert spacer == "ECK125136938"
    assert m10 == "9A->G/8A->T"
    assert bg == "bg463205:463355"


def test_parse_name_lac_spacer_truncated_no_bg_returns_none():
    name = "gourse_136fold_up_35T->C/33G->T/31G->C_lac-spacer"
    assert _parse_name(name) is None


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [f"v{i}" for i in range(100)],
            "sequence": ["ACGT" * 30] * 100,
            "expression": [0.1 * (i + 1) for i in range(100)],
            "log_expression": [-1.0] * 100,
            "up_class": ["noUP"] * 50 + ["gourse_136fold_up"] * 50,
            "m35_pattern": ["wt"] * 60 + ["35T->A"] * 40,
            "spacer": ["lac"] * 25 + ["ECK1"] * 25 + ["ECK2"] * 50,
            "m10_pattern": ["wt"] * 100,
            "background": ["bg1"] * 100,
        }
    )


def test_random_split_proportions():
    df = _toy_df()
    splits = random_split(df, val_frac=0.1, test_frac=0.2, seed=42)
    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 20
    total_names = (
        set(splits["train"]["name"])
        | set(splits["val"]["name"])
        | set(splits["test"]["name"])
    )
    assert total_names == set(df["name"])


def test_random_split_invalid_fractions():
    df = _toy_df()
    with pytest.raises(ValueError):
        random_split(df, val_frac=0.6, test_frac=0.5)


def test_leave_element_out_excludes_holdout():
    df = _toy_df()
    splits = leave_element_out_split(df, element_col="up_class", holdout_value="noUP")
    assert (splits["test"]["up_class"] == "noUP").all()
    assert (splits["train"]["up_class"] != "noUP").all()
    assert (splits["val"]["up_class"] != "noUP").all()


def test_leave_element_out_default_picks_largest():
    df = _toy_df()
    splits = leave_element_out_split(df, element_col="spacer")
    assert splits["holdout_value"] == "ECK2"
    assert (splits["test"]["spacer"] == "ECK2").all()


def test_leave_element_out_unknown_column_raises():
    df = _toy_df()
    with pytest.raises(KeyError):
        leave_element_out_split(df, element_col="nonexistent")
