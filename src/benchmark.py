"""Signal drift correction and missing-value imputation for untargeted
LC-MS metabolomics, benchmarked on simulated data with known truth.

Simulation (per study): 400 features, 100 study samples (50 cases, 50
controls) run in 4 analytical batches with a pooled QC sample injected at
the start of each batch and after every 5 study samples. Each feature has
  * a true biological abundance, with 40 features differing between groups
    by ±0.6 log2 units (about 1.5-fold)
  * a batch offset and a smooth within-batch drift over injection order
  * measurement noise
  * values below a feature-specific limit of detection reported as missing
    (missing not at random), plus 2% missing completely at random.

Drift correction: none; per-batch median centering; QC-based robust LOESS
signal correction (QC-RLSC), which fits each feature's drift on the pooled QC
injections within each batch and removes it from the study samples.

Imputation (after QC-RLSC): feature median; half of the feature minimum; k
nearest neighbours; a left-censored draw that imputes below the feature's
observed minimum, as missing-not-at-random data require; and a hybrid that
uses the left-censored draw for features missing in many samples and k
nearest neighbours for sporadic gaps.

Each method is scored on what it recovers: QC precision, agreement with the
true biological abundance, imputation error, and the false discovery
proportion and power of the downstream case-control test.

Usage: python src/benchmark.py [--studies 20]
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F, N, N_DE, BATCHES = 400, 100, 40, 4


def simulate(seed):
    rng = np.random.default_rng(seed)
    mu = rng.normal(16, 2, F)
    group = np.repeat([0, 1], N // 2)
    rng.shuffle(group)
    de = rng.choice(F, N_DE, replace=False)
    eff = np.zeros(F)
    eff[de] = rng.choice([-0.6, 0.6], N_DE)
    bio = mu[None, :] + rng.normal(0, 0.5, (N, F)) + group[:, None] * eff[None, :]
    qc_true = mu + 0.5 * eff                         # pooled QC: the average sample

    # Injection sequence: per batch, QC, then 5 samples and a QC, repeated.
    seq, s = [], 0
    per_batch = N // BATCHES
    for b in range(BATCHES):
        seq.append(("QC", b, None))
        for k in range(per_batch):
            seq.append(("S", b, s))
            s += 1
            if (k + 1) % 5 == 0:
                seq.append(("QC", b, None))
    order_in_batch = []
    for b in range(BATCHES):
        n_b = sum(1 for x in seq if x[1] == b)
        order_in_batch += list(np.linspace(0, 1, n_b))
    batch_off = rng.normal(0, 0.35, (BATCHES, F))
    a1, a2 = rng.normal(0, 0.5, (BATCHES, F)), rng.normal(0, 0.4, (BATCHES, F))
    lod = mu - rng.uniform(0.6, 1.4, F) * (mu < 15) - 3 * (mu >= 15)   # low-abundance features lose values

    rows, meta, why = [], [], []
    for (kind, b, i), t in zip(seq, order_in_batch):
        true = qc_true if kind == "QC" else bio[i]
        drift = batch_off[b] + a1[b] * t + a2[b] * (t - 0.5) ** 2
        obs = true + drift + rng.normal(0, 0.12, F)
        reason = np.where(obs < lod, 1, 0)                     # 1: below the limit of detection
        reason[(reason == 0) & (rng.random(F) < 0.02)] = 2     # 2: missing completely at random
        obs[reason > 0] = np.nan
        rows.append(obs)
        why.append(reason)
        meta.append({"type": kind, "batch": b, "order": t, "sample": i,
                     "group": None if kind == "QC" else int(group[i])})
    X = np.array(rows)
    meta = pd.DataFrame(meta)
    return X, meta, bio, eff, np.array(why)


def correct(X, meta, method):
    Y = X.copy()
    if method == "none":
        return Y
    for b in range(BATCHES):
        idx = np.flatnonzero(meta.batch == b)
        if method == "batch median":
            Y[idx] -= np.nanmedian(X[idx], axis=0) - np.nanmedian(X, axis=0)
            continue
        qc = idx[meta.type.values[idx] == "QC"]
        for f in range(X.shape[1]):
            ok = qc[~np.isnan(X[qc, f])]
            if len(ok) < 4:
                Y[idx, f] -= np.nanmedian(X[idx, f]) - np.nanmedian(X[:, f])
                continue
            fit = lowess(X[ok, f], meta.order.values[ok], frac=0.8, it=1, return_sorted=True)
            trend = np.interp(meta.order.values[idx], fit[:, 0], fit[:, 1])
            Y[idx, f] = X[idx, f] - trend + np.nanmedian(X[qc, f] if method == "QC-RLSC" else X[:, f])
    if method == "QC-RLSC":                                  # align batches on the overall QC median
        qc_all = meta.type.values == "QC"
        for b in range(BATCHES):
            idx = np.flatnonzero(meta.batch == b)
            qb = idx[meta.type.values[idx] == "QC"]
            Y[idx] -= np.nanmedian(Y[qb], axis=0) - np.nanmedian(Y[qc_all], axis=0)
    return Y


def impute(S, method, rng):
    S = S.copy()
    miss = np.isnan(S)
    if method == "feature median":
        S[miss] = np.take(np.nanmedian(S, axis=0), np.where(miss)[1])
    elif method == "half minimum":
        S[miss] = np.take(np.nanmin(S, axis=0) - 1.0, np.where(miss)[1])     # half the minimum, on the log2 scale
    elif method == "k nearest neighbours":
        S = KNNImputer(n_neighbors=10).fit_transform(S)
    elif method == "left-censored draw":
        mn, sd = np.nanmin(S, axis=0), np.nanstd(S, axis=0)
        for f in np.flatnonzero(miss.any(axis=0)):
            k = miss[:, f].sum()
            # draw from the lower tail: a normal truncated above at the observed minimum
            lo, hi = -np.inf, 0.0
            S[miss[:, f], f] = mn[f] + sd[f] * 0.5 * stats.truncnorm.rvs(lo, hi, size=k, random_state=rng)
    elif method == "hybrid by missingness pattern":
        # Features missing in many samples are most likely censored at the detection limit:
        # impute those from the lower tail; impute sporadic gaps from similar samples.
        frac = miss.mean(axis=0)
        censored = frac >= 0.08
        L = impute(S, "left-censored draw", rng)
        K = impute(S, "k nearest neighbours", rng)
        S = np.where(censored[None, :], L, K)
    return S


def de_test(S, group, eff):
    p = stats.ttest_ind(S[group == 1], S[group == 0], axis=0, equal_var=False, nan_policy="omit").pvalue
    p = np.nan_to_num(p, nan=1.0)
    rej = multipletests(p, 0.05, method="fdr_bh")[0]
    truth = eff != 0
    return rej.sum(), (rej & ~truth).sum() / max(rej.sum(), 1), (rej & truth).sum() / truth.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=20)
    a = ap.parse_args()
    corr_rows, imp_rows = [], []
    example = None
    for st in range(a.studies):
        X, meta, bio, eff, why = simulate(st)
        smp = (meta.type == "S").values
        order = meta.loc[smp, "sample"].astype(int).values
        group = meta.loc[smp, "group"].astype(int).values
        truth_s = bio[order]
        for m in ("none", "batch median", "QC-RLSC"):
            Y = correct(X, meta, m)
            qc = Y[meta.type.values == "QC"]
            qc_sd = np.nanstd(qc, axis=0)
            S = Y[smp]
            r = [stats.pearsonr(S[~np.isnan(S[:, f]), f], truth_s[~np.isnan(S[:, f]), f])[0]
                 for f in range(F) if (~np.isnan(S[:, f])).sum() > 10]
            n, fdp, power = de_test(S, group, eff)
            corr_rows.append({"study": st, "correction": m, "median_qc_sd_log2": np.nanmedian(qc_sd),
                              "share_features_qc_cv_below_20pct": np.mean(qc_sd * np.log(2) < 0.20),
                              "median_r_with_true_abundance": np.median(r), "fdp": fdp, "power": power})
            if m == "QC-RLSC" and st == 0:
                example = (X, Y, meta)
        S = correct(X, meta, "QC-RLSC")[smp]
        miss = np.isnan(S)
        below = why[smp] == 1
        mcar = why[smp] == 2
        rng = np.random.default_rng(st)
        # Imputation error is measured against the value the corrected data would hold had it been
        # detected: the true abundance plus each feature's offset between corrected and true scales,
        # estimated from the paired observed values (so censoring does not bias the offset).
        target = truth_s + np.nanmedian(S - truth_s, axis=0)[None, :]
        for m in ("feature median", "half minimum", "k nearest neighbours", "left-censored draw",
                  "hybrid by missingness pattern"):
            I = impute(S, m, rng)
            err = I - target
            n, fdp, power = de_test(I, group, eff)
            imp_rows.append({"study": st, "imputation": m, "share_missing": miss.mean(),
                             "share_missing_below_lod": below.sum() / miss.sum(),
                             "rmse_below_lod": np.sqrt(np.mean(err[below] ** 2)), "bias_below_lod": err[below].mean(),
                             "rmse_random_missing": np.sqrt(np.mean(err[mcar] ** 2)),
                             "bias_random_missing": err[mcar].mean(), "fdp": fdp, "power": power})
        print(f"study {st + 1} done", flush=True)

    c = pd.DataFrame(corr_rows).groupby("correction", sort=False).mean(numeric_only=True).drop(columns="study").round(3)
    i = pd.DataFrame(imp_rows).groupby("imputation", sort=False).mean(numeric_only=True).drop(columns="study").round(3)
    c.to_csv(os.path.join(ROOT, "results", "tables", "drift_correction.csv"))
    i.to_csv(os.path.join(ROOT, "results", "tables", "imputation.csv"))

    X, Y, meta = example
    f = int(np.nanargmax(np.nanstd(X[meta.type.values == "QC"], axis=0) * (np.isnan(X).sum(0) == 0)))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    x = np.arange(len(meta))
    for k, (M, title) in enumerate([(X, "before correction"), (Y, "after QC-RLSC")]):
        for kind, col in (("S", "#888"), ("QC", "#d62728")):
            sel = meta.type.values == kind
            ax[k].scatter(x[sel], M[sel, f], s=10, color=col, label="pooled QC" if kind == "QC" else "study sample")
        for b in range(1, BATCHES):
            ax[k].axvline(np.flatnonzero(meta.batch == b)[0] - 0.5, color="k", lw=0.6, ls=":")
        ax[k].set_title(f"Feature {f}, {title}")
        ax[k].set_xlabel("Injection order (dotted lines: batches)")
    ax[0].set_ylabel("log2 intensity")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "drift_example.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    w = 0.38
    xx = np.arange(len(i))
    ax[0].bar(xx - w / 2, i.rmse_below_lod, w, label="below limit of detection", color="#1f77b4")
    ax[0].bar(xx + w / 2, i.rmse_random_missing, w, label="missing at random", color="#aec7e8")
    ax[0].set_xticks(xx, i.index)
    ax[0].legend(fontsize=8)
    ax[0].set_ylabel("RMSE of imputed values (log2)")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(i.index, i.fdp, color="#d62728")
    ax[1].axhline(0.05, ls=":", color="k")
    ax[1].set_ylabel("False discovery proportion (target 5%)")
    ax[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Imputation of values missing below the limit of detection")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "imputation.png"), dpi=150)
    plt.close(fig)
    print(c.to_string())
    print(i.to_string())


if __name__ == "__main__":
    main()
