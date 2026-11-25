
"""Population Stability Index and a drift report. You write this file."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(reference, current, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index between two 1-D numeric samples."""
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()

    # Drop NaNs from both samples.
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if ref.size == 0 or cur.size == 0:
        raise ValueError("Reference and current samples must both contain values.")

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    if eps <= 0:
        raise ValueError("eps must be greater than 0.")

    # Calculate bin edges using the reference distribution only.
    edges = np.unique(
        np.quantile(ref, np.linspace(0, 1, n_bins + 1))[1:-1]
    )

    # Assign values to bins.
    ref_bins = np.searchsorted(edges, ref, side="right")
    cur_bins = np.searchsorted(edges, cur, side="right")

    n_actual_bins = len(edges) + 1

    # Calculate proportions for each bin.
    ref_counts = np.bincount(ref_bins, minlength=n_actual_bins)
    cur_counts = np.bincount(cur_bins, minlength=n_actual_bins)

    ref_prop = ref_counts / ref.size
    cur_prop = cur_counts / cur.size

    # Apply the minimum proportion and calculate PSI.
    ref_prop = np.maximum(ref_prop, eps)
    cur_prop = np.maximum(cur_prop, eps)

    return float(np.sum(
        (cur_prop - ref_prop) * np.log(cur_prop / ref_prop)
    ))


def categorical_psi(reference, current, eps: float = 1e-4) -> float:
    """PSI for a categorical column."""
    ref = pd.Series(reference).dropna().astype(str).to_numpy()
    cur = pd.Series(current).dropna().astype(str).to_numpy()

    if ref.size == 0 or cur.size == 0:
        raise ValueError("Reference and current samples must both contain values.")

    if eps <= 0:
        raise ValueError("eps must be greater than 0.")

    # Include categories found on either side.
    categories = sorted(set(ref) | set(cur))

    ref_counts = np.array([
        np.count_nonzero(ref == category) for category in categories
    ], dtype=float)

    cur_counts = np.array([
        np.count_nonzero(cur == category) for category in categories
    ], dtype=float)

    ref_prop = ref_counts / ref.size
    cur_prop = cur_counts / cur.size

    ref_prop = np.maximum(ref_prop, eps)
    cur_prop = np.maximum(cur_prop, eps)

    return float(np.sum(
        (cur_prop - ref_prop) * np.log(cur_prop / ref_prop)
    ))


def drift_report(
    ref_df,
    cur_df,
    numeric: list[str],
    categorical: list[str],
    threshold: float = 0.2,
) -> dict:
    """Generate a drift report for numeric and categorical columns."""
    columns = {}
    drifted_columns = []

    for col in numeric + categorical:
        ref_series = ref_df[col]
        cur_series = cur_df[col]

        # Calculate missing rates using the original data.
        missing_rate_ref = float(ref_series.isna().mean())
        missing_rate_cur = float(cur_series.isna().mean())

        # Calculate PSI according to the column type.
        if col in numeric:
            score = psi(ref_series, cur_series)
        else:
            score = categorical_psi(ref_series, cur_series)

        drifted = (
            score >= threshold
            or abs(missing_rate_cur - missing_rate_ref) >= 0.1
        )

        columns[col] = {
            "psi": score,
            "missing_rate_ref": missing_rate_ref,
            "missing_rate_cur": missing_rate_cur,
            "drifted": bool(drifted),
        }

        if drifted:
            drifted_columns.append(col)

    return {
        "columns": columns,
        "drifted_columns": sorted(drifted_columns),
    }
