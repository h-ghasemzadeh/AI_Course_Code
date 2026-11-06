#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distribution fitting and goodness-of-fit analysis for civil-engineering data.
Example dataset: UCI Concrete Compressive Strength (Yeh, 1998), 1,030 records.

Usage:
  pip install numpy pandas scipy matplotlib openpyxl ucimlrepo
  python distribution_analysis.py
  python distribution_analysis.py --file Concrete_Data.xls --column "Concrete compressive strength"
  python distribution_analysis.py --file my_data.csv --column strength_mpa

Notes:
- Fits continuous distributions to the selected numeric variable.
- Binomial and Poisson are discrete distributions; they are analyzed separately
  after rounding the selected measurements to integers, and are not directly
  comparable to continuous distributions fitted to unrounded MPa data.
- For continuous distributions, KS and Anderson-Darling p-values use SciPy's
  parametric-bootstrap goodness_of_fit when available (parameters are refitted
  in each Monte Carlo sample). The grouped chi-square test is approximate.
- Passing a goodness-of-fit test does NOT prove a distribution is true. Compare
  p-values together with AIC/BIC, QQ plots, physical plausibility, and sample size.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

CONTINUOUS = {
    "Normal": stats.norm,
    "Lognormal": stats.lognorm,
    "Gamma": stats.gamma,
    "Beta": stats.beta,
    "Chi-square": stats.chi2,
    "Exponential": stats.expon,
}

def load_data(file_path, column):
    if file_path:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        if p.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p)
    else:
        # Fetch the public UCI dataset. Requires internet access on first run.
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as e:
            raise RuntimeError(
                "Install ucimlrepo or download the UCI Excel file and use --file. "
                "Install with: pip install ucimlrepo"
            ) from e
        ds = fetch_ucirepo(id=165)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        # normalize the target header from repository metadata
        if column is None:
            column = ds.data.targets.columns[0]
    if column is None:
        print("\nColumns in file:")
        for c in df.columns:
            print(" -", repr(c))
        raise ValueError("Pass the desired numeric column with --column.")
    if column not in df.columns:
        raise ValueError(
            f"Column {column!r} not found. Available columns: {list(df.columns)}"
        )
    x = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        raise ValueError("At least 8 finite numeric observations are required.")
    return df, x, column

def describe(x):
    mean = np.mean(x)
    var = np.var(x, ddof=1)
    sd = np.std(x, ddof=1)
    skew = stats.skew(x, bias=False)
    cv = sd / abs(mean) * 100 if mean != 0 else np.nan
    return {
        "n": len(x), "min": np.min(x), "max": np.max(x),
        "mean": mean, "variance_sample": var, "std_sample": sd,
        "skewness": skew, "coefficient_of_variation_percent": cv,
        "median": np.median(x), "q1": np.quantile(x, .25),
        "q3": np.quantile(x, .75), "excess_kurtosis": stats.kurtosis(x, bias=False),
    }

def fit_continuous(x):
    results = []
    fitted = {}
    positive = np.all(x > 0)
    nonconstant = np.ptp(x) > 0
    for name, dist in CONTINUOUS.items():
        if not nonconstant:
            continue
        if name in ("Lognormal", "Gamma", "Beta", "Chi-square", "Exponential") and not positive:
            results.append({"distribution": name, "status": "not fitted: requires positive data"})
            continue
        try:
            # Beta is fitted on the observed range; this estimates loc/scale too.
            params = dist.fit(x)
            logpdf = dist.logpdf(x, *params)
            if not np.all(np.isfinite(logpdf)):
                raise ValueError("non-finite log likelihood")
            ll = float(np.sum(logpdf))
            k = len(params)
            aic = 2*k - 2*ll
            bic = k*np.log(len(x)) - 2*ll
            fitted[name] = (dist, params)
            row = {"distribution": name, "status": "ok", "parameters": repr(tuple(float(p) for p in params)),
                   "log_likelihood": ll, "AIC": aic, "BIC": bic,
                   "KS_D": np.nan, "KS_p_bootstrap": np.nan,
                   "AD_statistic": np.nan, "AD_p_bootstrap": np.nan,
                   "chi_square": np.nan, "chi_square_df": np.nan, "chi_square_p": np.nan}
            # Proper Monte-Carlo GOF with parameter refitting, when supported by SciPy.
            if hasattr(stats, "goodness_of_fit"):
                try:
                    ks = stats.goodness_of_fit(dist, x, statistic="ks",
                                               n_mc_samples=499, rng=np.random.default_rng(2026))
                    ad = stats.goodness_of_fit(dist, x, statistic="ad",
                                               n_mc_samples=499, rng=np.random.default_rng(2027))
                    row["KS_D"] = float(ks.statistic)
                    row["KS_p_bootstrap"] = float(ks.pvalue)
                    row["AD_statistic"] = float(ad.statistic)
                    row["AD_p_bootstrap"] = float(ad.pvalue)
                except Exception as e:
                    row["GOF_note"] = f"bootstrap test unavailable for this fit: {e}"
            # Approximate grouped chi-square test: choose quantile bins with equal expected counts.
            try:
                n = len(x)
                # Quantile bins avoid tiny expected counts in tails; merge is unnecessary
                # because each theoretical bin has expected count n/10 by construction.
                probs = np.linspace(0, 1, 11)
                edges = dist.ppf(probs, *params)
                edges[0], edges[-1] = -np.inf, np.inf
                edges = np.unique(edges)
                if len(edges) >= 4:
                    observed, _ = np.histogram(x, bins=edges)
                    cdf_edges = dist.cdf(edges, *params)
                    expected = n * np.diff(cdf_edges)
                    mask = expected > 1e-8
                    observed, expected = observed[mask], expected[mask]
                    # number of fitted parameters estimated from the same data
                    dfree = len(observed) - 1 - k
                    if dfree > 0 and np.all(expected >= 5):
                        stat = float(np.sum((observed-expected)**2/expected))
                        row["chi_square"] = stat
                        row["chi_square_df"] = int(dfree)
                        row["chi_square_p"] = float(stats.chi2.sf(stat, dfree))
            except Exception:
                pass
            results.append(row)
        except Exception as e:
            results.append({"distribution": name, "status": f"fit failed: {e}"})
    table = pd.DataFrame(results)
    if "AIC" in table:
        table = table.sort_values("AIC", na_position="last")
    return table, fitted

def discrete_fits(x):
    # Discrete models require integer-valued observations. For MPa strengths, this
    # is an explicitly separate analysis of rounded values, not the original data.
    xr = np.rint(x).astype(int)
    rows = []
    if np.any(xr < 0):
        return pd.DataFrame([{"distribution": "Binomial / Poisson",
                              "status": "not fitted: rounded observations include negative values"}])
    nobs = len(xr)
    # Binomial: choose n as the largest observed integer and estimate p by mean/n.
    n_trials = int(max(1, np.max(xr)))
    p_hat = float(np.clip(np.mean(xr) / n_trials, 1e-8, 1-1e-8))
    # Binomial fit may be structurally inappropriate for strength measurements.
    for name, dist, params, k in [
        ("Binomial (rounded data)", stats.binom, (n_trials, p_hat), 2),
        ("Poisson (rounded data)", stats.poisson, (float(np.mean(xr)),), 1),
    ]:
        try:
            ll = float(np.sum(dist.logpmf(xr, *params)))
            rows.append({"distribution": name, "parameters": repr(params),
                         "log_likelihood": ll, "AIC": 2*k-2*ll,
                         "BIC": k*np.log(nobs)-2*ll,
                         "note": "rounded integers; compare only with discrete models"})
        except Exception as e:
            rows.append({"distribution": name, "status": f"failed: {e}"})
    return pd.DataFrame(rows).sort_values("AIC", na_position="last")

def normality_tests(x):
    out = []
    # These tests specifically assess normality; they do not test every candidate distribution.
    tests = [
        ("Shapiro-Wilk", lambda: stats.shapiro(x)),
        ("D'Agostino-Pearson", lambda: stats.normaltest(x)),
        ("Jarque-Bera", lambda: stats.jarque_bera(x)),
        ("Anderson-Darling normality", lambda: stats.anderson(x, dist="norm")),
    ]
    for name, fn in tests:
        try:
            r = fn()
            if name == "Anderson-Darling normality":
                # scipy.anderson returns critical values, not a p-value.
                out.append({"test": name, "statistic": float(r.statistic),
                            "p_value": np.nan,
                            "note": "SciPy returns critical values, not a p-value; compare statistic to critical_values"})
            else:
                out.append({"test": name, "statistic": float(r.statistic),
                            "p_value": float(r.pvalue), "note": "tests normality only"})
        except Exception as e:
            out.append({"test": name, "note": str(e)})
    return pd.DataFrame(out)

def plots(x, column, fitted, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    # 1. Raw observations and histogram
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(np.arange(1, len(x)+1), x, ".", ms=3, alpha=.65)
    ax[0].set(title=f"Observed data: {column}", xlabel="Observation index", ylabel=column)
    ax[0].grid(alpha=.25)
    ax[1].hist(x, bins="fd", density=True, alpha=.55, edgecolor="black", label="Empirical density")
    xx = np.linspace(np.min(x), np.max(x), 600)
    for name, (dist, params) in fitted.items():
        try:
            yy = dist.pdf(xx, *params)
            if np.all(np.isfinite(yy)):
                ax[1].plot(xx, yy, lw=1.6, label=name)
        except Exception:
            pass
    ax[1].set(title="Histogram and fitted PDFs", xlabel=column, ylabel="Density")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(outdir / "01_data_and_histogram.png", dpi=220)
    plt.close(fig)

    # 2. QQ plots: theoretical quantiles against sample quantiles
    nplots = len(fitted)
    if nplots:
        ncols = 3
        nrows = int(np.ceil(nplots / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows))
        axes = np.array(axes).reshape(-1)
        probs = (np.arange(1, len(x)+1) - .5) / len(x)
        xs = np.sort(x)
        for ax, (name, (dist, params)) in zip(axes, fitted.items()):
            tq = dist.ppf(probs, *params)
            ax.scatter(tq, xs, s=7, alpha=.55)
            lo = np.nanmin([np.nanmin(tq), np.nanmin(xs)])
            hi = np.nanmax([np.nanmax(tq), np.nanmax(xs)])
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            ax.set_title(f"QQ plot: {name}")
            ax.set_xlabel("Theoretical quantiles")
            ax.set_ylabel("Observed quantiles")
            ax.grid(alpha=.2)
        for ax in axes[nplots:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(outdir / "02_qq_plots.png", dpi=220)
        plt.close(fig)

    # 3. Empirical CDF and fitted CDFs
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.sort(x)
    ax.step(xs, np.arange(1, len(xs)+1)/len(xs), where="post", label="Empirical CDF", lw=2)
    grid = np.linspace(np.min(x), np.max(x), 700)
    for name, (dist, params) in fitted.items():
        ax.plot(grid, dist.cdf(grid, *params), label=name, lw=1.5)
    ax.set(title="Empirical and fitted cumulative distributions", xlabel=column, ylabel="CDF")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(outdir / "03_ecdf_fitted_cdfs.png", dpi=220)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Fit distributions and run goodness-of-fit tests.")
    parser.add_argument("--file", help="CSV/XLS/XLSX input file. Omit to fetch UCI concrete data.")
    parser.add_argument("--column", help="Numeric column name to analyze.")
    parser.add_argument("--outdir", default="distribution_results", help="Output directory.")
    args = parser.parse_args()

    df, x, column = load_data(args.file, args.column)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    desc = pd.DataFrame([describe(x)])
    cont_table, fitted = fit_continuous(x)
    disc_table = discrete_fits(x)
    norm_table = normality_tests(x)

    desc.to_csv(outdir / "descriptive_statistics.csv", index=False)
    cont_table.to_csv(outdir / "continuous_distribution_fits.csv", index=False)
    disc_table.to_csv(outdir / "discrete_distribution_fits_rounded_data.csv", index=False)
    norm_table.to_csv(outdir / "normality_tests.csv", index=False)
    plots(x, column, fitted, outdir)

    print("\n=== Descriptive statistics ===")
    print(desc.to_string(index=False))
    print("\n=== Continuous distribution fits (sorted by AIC) ===")
    print(cont_table.to_string(index=False))
    print("\n=== Discrete models on rounded data (separate analysis) ===")
    print(disc_table.to_string(index=False))
    print("\n=== Normality tests ===")
    print(norm_table.to_string(index=False))

    successful = cont_table[cont_table.get("status", pd.Series(index=cont_table.index, dtype=object)) == "ok"].copy()
    if not successful.empty:
        print("\nInterpretation guide:")
        print("- Smaller AIC/BIC indicates a better relative fit among the candidate models.")
        print("- Larger bootstrap KS/AD p-values mean less evidence against that fitted model; they do not prove it is true.")
        print("- Check QQ plots, physical support, and whether tails are modeled well.")
        print("- Do not select a distribution by p-value alone.")
        print(f"\nLowest AIC candidate: {successful.sort_values('AIC').iloc[0]['distribution']}")
    print(f"\nAll tables and figures saved to: {outdir.resolve()}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
