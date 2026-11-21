"""Sanity checks on the benchmark outputs and on the correction itself."""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from benchmark import correct, simulate  # noqa: E402

TAB = os.path.join(ROOT, "results", "tables")


def test_qc_rlsc_tightens_pooled_qc():
    X, meta, *_ = simulate(123)
    qc = meta.type.values == "QC"
    before = np.nanmedian(np.nanstd(X[qc], axis=0))
    after = np.nanmedian(np.nanstd(correct(X, meta, "QC-RLSC")[qc], axis=0))
    assert after < 0.5 * before


def test_correction_ranking():
    c = pd.read_csv(os.path.join(TAB, "drift_correction.csv")).set_index("correction")
    assert c.loc["QC-RLSC", "median_r_with_true_abundance"] > c.loc["batch median", "median_r_with_true_abundance"] \
        > c.loc["none", "median_r_with_true_abundance"]


def test_imputation_by_mechanism():
    i = pd.read_csv(os.path.join(TAB, "imputation.csv")).set_index("imputation")
    assert i.rmse_below_lod.idxmin() == "left-censored draw"
    assert i.loc["feature median", "rmse_random_missing"] < i.loc["left-censored draw", "rmse_random_missing"]
    assert i.loc["half minimum"].rmse_below_lod > i.loc["left-censored draw"].rmse_below_lod
