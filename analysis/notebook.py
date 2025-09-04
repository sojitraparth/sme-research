
"""
SME Research – Analysis Toolkit
Reads results/benchmarks.csv and produces figure-ready tables:
- results/derived/medians.csv
- results/derived/effects_hedges_g.csv
- results/derived/effects_cliffs_delta.csv
- results/derived/effects_common_language.csv
- results/derived/anova_tukey.txt   (if statsmodels present)

Also prints a compact console report.
"""

from __future__ import annotations
import math, itertools, json
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

BENCH = Path("results/benchmarks.csv")
OUTDIR = Path("results/derived")
OUTDIR.mkdir(parents=True, exist_ok=True)

def pooled_sd(x: np.ndarray, y: np.ndarray) -> float:
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1 = np.var(x, ddof=1)
    s2 = np.var(y, ddof=1)
    df = (n1 - 1) + (n2 - 1)
    if df <= 0:
        return float("nan")
    sp2 = ((n1 - 1) * s1 + (n2 - 1) * s2) / df
    return math.sqrt(sp2)

def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased standardized mean diff with Hedges' J correction."""
    n1, n2 = len(x), len(y)
    sp = pooled_sd(x, y)
    if not np.isfinite(sp) or sp == 0:
        return float("nan")
    d = (np.mean(x) - np.mean(y)) / sp
    df = n1 + n2 - 2
    J = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 0 else 1.0  # Hedges' correction
    return J * d

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """δ = P(x>y) - P(x<y), robust, non-parametric."""
    # n and m are small (reps), so an O(n*m) loop is fine and clearer.
    gt = lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    n_pairs = len(x) * len(y)
    if n_pairs == 0:
        return float("nan")
    return (gt - lt) / n_pairs

def common_language_es(x: np.ndarray, y: np.ndarray) -> float:
    """CLES = P(x>y) using brute force (sample-level probability)."""
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = 0
    for xi in x:
        gt += np.sum(xi > y)
    return gt / (len(x) * len(y))

def _pairs(seq: List[str]) -> List[Tuple[str, str]]:
    return [p for p in itertools.combinations(sorted(seq), 2)]

def _series(df: pd.DataFrame, engine: str) -> np.ndarray:
    s = df.loc[df["engine"] == engine, "time_s"].dropna().to_numpy()
    return s.astype(float)

def main() -> None:
    if not BENCH.exists():
        print("No results/benchmarks.csv found. Run a benchmark first.")
        return
    df = pd.read_csv(BENCH)
    # Basic hygiene
    df = df.dropna(subset=["engine", "task", "time_s"])
    df["engine"] = df["engine"].astype(str)
    df["task"] = df["task"].astype(str)

    # 1) Medians by engine × task
    med = (
        df.groupby(["task", "engine"], as_index=False)
          .agg(n=("time_s", "size"),
               median_time_s=("time_s", "median"),
               iqr_time_s=("time_s", lambda s: s.quantile(0.75) - s.quantile(0.25)),
               median_rss_mb=("rss_mb", "median"))
          .sort_values(["task", "median_time_s"])
    )
    med.to_csv(OUTDIR / "medians.csv", index=False)

    # 2) Pairwise effect sizes within each task
    rows_g, rows_delta, rows_cles = [], [], []
    for task, gdf in df.groupby("task"):
        engines = sorted(gdf["engine"].unique())
        for a, b in _pairs(engines):
            xa = _series(gdf, a)
            xb = _series(gdf, b)
            rows_g.append({
                "task": task, "A": a, "B": b,
                "hedges_g": hedges_g(xa, xb),
                "A_mean": np.mean(xa) if xa.size else np.nan,
                "B_mean": np.mean(xb) if xb.size else np.nan,
                "A_n": xa.size, "B_n": xb.size
            })
            dlt = cliffs_delta(xa, xb)
            rows_delta.append({"task": task, "A": a, "B": b, "cliffs_delta": dlt})
            cles = common_language_es(xa, xb)
            rows_cles.append({"task": task, "A": a, "B": b, "cles_P(A>B)": cles})

    eff_g = pd.DataFrame(rows_g)
    eff_delta = pd.DataFrame(rows_delta)
    eff_cles = pd.DataFrame(rows_cles)

    eff_g.to_csv(OUTDIR / "effects_hedges_g.csv", index=False)
    eff_delta.to_csv(OUTDIR / "effects_cliffs_delta.csv", index=False)
    eff_cles.to_csv(OUTDIR / "effects_common_language.csv", index=False)

    # 3) Optional: one-way ANOVA per task + Tukey/Games–Howell (if statsmodels available)
    anova_report = []
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        HAVE_SM = True
    except Exception as e:
        HAVE_SM = False
        anova_report.append(f"[info] statsmodels not available: {e}")

    if HAVE_SM:
        for task, gdf in df.groupby("task"):
            # One-way ANOVA (engine -> time_s)
            try:
                model = ols("time_s ~ C(engine)", data=gdf).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                anova_report.append(f"\n=== Task: {task} — One-way ANOVA (time_s ~ engine) ===\n{anova_table}")
            except Exception as e:
                anova_report.append(f"\n=== Task: {task} — ANOVA failed: {e}")

            # Tukey HSD assuming equal variances (use Games–Howell if unequal; requires newer statsmodels)
            try:
                thsd = pairwise_tukeyhsd(endog=gdf["time_s"].to_numpy(),
                                         groups=gdf["engine"].astype(str).to_numpy(),
                                         alpha=0.05,
                                         use_var="equal")   # if variances unequal, consider "unequal" (Games–Howell)
                anova_report.append(str(thsd.summary()))
            except Exception as e:
                anova_report.append(f"Tukey/Games-Howell failed: {e}")

        (OUTDIR / "anova_tukey.txt").write_text("\n".join(anova_report), encoding="utf-8")
    else:
        (OUTDIR / "anova_tukey.txt").write_text("\n".join(anova_report), encoding="utf-8")

    # 4) Console summary
    print("\n# SME Research — Analysis Summary")
    print(f"Rows: {len(df):,} | Datasets: {df['dataset'].nunique()} | Tasks: {df['task'].nunique()} | Engines: {df['engine'].nunique()}")
    print("\n## Median time by engine × task (first 12 rows):")
    print(med.head(12).to_string(index=False))

    print("\n## Hedges' g by task (A faster than B ⇒ negative g when using time_s):")
    print(eff_g.head(12).to_string(index=False))

    print("\n## Cliff's delta by task (δ<0 ⇒ A usually faster than B):")
    print(eff_delta.head(12).to_string(index=False))

    if HAVE_SM:
        print("\n[statsmodels] ANOVA/Tukey written to results/derived/anova_tukey.txt")
    else:
        print("\n[info] Install statsmodels for ANOVA/Tukey: pip install statsmodels scipy")
    print("\nOutputs written to results/derived/*.csv")
    print("Done.")
    
if __name__ == "__main__":
    main()
