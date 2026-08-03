from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        value = float(values[0]) if len(values) else np.nan
        return value, value
    mean = float(np.mean(values))
    sem = stats.sem(values)
    critical = stats.t.ppf((1 + confidence) / 2, df=len(values) - 1)
    margin = float(critical * sem)
    return mean - margin, mean + margin


def cohen_dz(first: np.ndarray, second: np.ndarray) -> float:
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    sd = differences.std(ddof=1)
    return float(differences.mean() / sd) if sd > 0 else 0.0


def paired_tests(
    frame: pd.DataFrame,
    *,
    condition_column: str,
    pair_column: str,
    value_column: str,
    conditions: list[str] | None = None,
) -> pd.DataFrame:
    #Paired t-tests and Wilcoxon tests with Holm correction.

    if conditions is None:
        conditions = sorted(frame[condition_column].dropna().unique().tolist())

    rows = []
    for first, second in combinations(conditions, 2):
        first_values = frame.loc[frame[condition_column] == first, [pair_column, value_column]].rename(
            columns={value_column: "first"}
        )
        second_values = frame.loc[frame[condition_column] == second, [pair_column, value_column]].rename(
            columns={value_column: "second"}
        )
        merged = first_values.merge(second_values, on=pair_column, how="inner").dropna()
        a = merged["first"].to_numpy(float)
        b = merged["second"].to_numpy(float)
        if len(a) < 2:
            continue

        differences = a - b
        if np.allclose(differences, 0.0, atol=1e-12, rtol=1e-9):
            t_stat, t_p = 0.0, 1.0
            w_stat, w_p = 0.0, 1.0
        else:
            t_result = stats.ttest_rel(a, b, nan_policy="omit")
            t_stat = float(t_result.statistic) if np.isfinite(t_result.statistic) else np.nan
            t_p = float(t_result.pvalue) if np.isfinite(t_result.pvalue) else np.nan
            try:
                w_result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                w_stat = float(w_result.statistic)
                w_p = float(w_result.pvalue)
            except ValueError:
                w_stat, w_p = 0.0, 1.0

        rows.append(
            {
                "condition_a": first,
                "condition_b": second,
                "n_pairs": int(len(a)),
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "mean_difference_a_minus_b": float(differences.mean()),
                "cohen_dz": cohen_dz(a, b),
                "paired_t_statistic": t_stat,
                "paired_t_p": t_p,
                "wilcoxon_statistic": w_stat,
                "wilcoxon_p": w_p,
            }
        )

    result = pd.DataFrame(rows)
    if len(result):
        result["paired_t_p_holm"] = multipletests(result["paired_t_p"], method="holm")[1]
        result["wilcoxon_p_holm"] = multipletests(result["wilcoxon_p"], method="holm")[1]
    return result


def friedman_test(
    frame: pd.DataFrame,
    *,
    condition_column: str,
    pair_column: str,
    value_column: str,
    conditions: list[str] | None = None,
) -> pd.DataFrame:
    if conditions is None:
        conditions = sorted(frame[condition_column].dropna().unique().tolist())
    pivot = frame.pivot_table(index=pair_column, columns=condition_column, values=value_column)
    pivot = pivot.dropna(subset=conditions)
    if len(pivot) < 2 or len(conditions) < 3:
        return pd.DataFrame(
            [{"metric": value_column, "n_pairs": len(pivot), "statistic": np.nan, "p_value": np.nan}]
        )
    arrays = [pivot[c].to_numpy(float) for c in conditions]
    if all(np.allclose(arrays[0], other, atol=1e-12, rtol=1e-9) for other in arrays[1:]):
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.friedmanchisquare(*arrays)
        statistic = float(result.statistic) if np.isfinite(result.statistic) else np.nan
        p_value = float(result.pvalue) if np.isfinite(result.pvalue) else np.nan
    return pd.DataFrame(
        [
            {
                "metric": value_column,
                "n_pairs": int(len(pivot)),
                "statistic": statistic,
                "p_value": p_value,
            }
        ]
    )


def aggregate_seed_metrics(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    metric_columns: list[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys, strict=True))
        for metric in metric_columns:
            values = group[metric].to_numpy(float)
            low, high = confidence_interval(values, confidence=confidence)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_seeds": int(len(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)
