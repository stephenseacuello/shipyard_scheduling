"""Statistical analysis utilities for publication-quality results.

This module provides functions for computing confidence intervals,
significance tests, effect sizes, and generating publication-ready
result tables with proper statistical formatting.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy import stats


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute mean and confidence interval for data.

    Uses Student's t-distribution for small sample sizes.

    Args:
        data: Array of observations.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    n = len(data)
    if n < 2:
        mean = float(np.mean(data))
        return mean, mean, mean

    mean = float(np.mean(data))
    se = stats.sem(data)
    ci_half = se * stats.t.ppf((1 + confidence) / 2, n - 1)

    return mean, mean - ci_half, mean + ci_half


def compute_std_error(data: np.ndarray) -> float:
    """Compute standard error of the mean.

    Args:
        data: Array of observations.

    Returns:
        Standard error.
    """
    return float(stats.sem(data))


def paired_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Perform paired t-test for comparing two related samples.

    Appropriate when comparing the same subjects under two conditions
    (e.g., same random seeds, different algorithms).

    Args:
        group1: First group of observations.
        group2: Second group of observations (paired with group1).
        alternative: One of 'two-sided', 'less', or 'greater'.

    Returns:
        Dictionary with t_stat, p_value, and effect size (Cohen's d).
    """
    if len(group1) != len(group2):
        raise ValueError("Groups must have same length for paired test")

    t_stat, p_value = stats.ttest_rel(group1, group2, alternative=alternative)

    # Cohen's d for paired samples
    diff = group1 - group2
    cohens_d = float(np.mean(diff) / np.std(diff, ddof=1))

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
        "n": len(group1),
    }


def independent_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = "two-sided",
    equal_var: bool = False,
) -> Dict[str, float]:
    """Perform independent t-test for comparing two unrelated samples.

    Uses Welch's t-test by default (does not assume equal variances).

    Args:
        group1: First group of observations.
        group2: Second group of observations.
        alternative: One of 'two-sided', 'less', or 'greater'.
        equal_var: If True, assume equal population variances.

    Returns:
        Dictionary with t_stat, p_value, and effect size (Cohen's d).
    """
    t_stat, p_value = stats.ttest_ind(
        group1, group2, alternative=alternative, equal_var=equal_var
    )

    # Cohen's d for independent samples (pooled std)
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = float((np.mean(group1) - np.mean(group2)) / pooled_std)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
        "n1": n1,
        "n2": n2,
    }


def mann_whitney_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Perform Mann-Whitney U test (non-parametric).

    Use when normality assumption is violated.

    Args:
        group1: First group of observations.
        group2: Second group of observations.
        alternative: One of 'two-sided', 'less', or 'greater'.

    Returns:
        Dictionary with statistic, p_value, and effect size (rank-biserial r).
    """
    stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)

    # Rank-biserial correlation as effect size
    n1, n2 = len(group1), len(group2)
    r = 1 - (2 * stat) / (n1 * n2)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "rank_biserial_r": float(r),
        "n1": n1,
        "n2": n2,
    }


def wilcoxon_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Perform Wilcoxon signed-rank test (non-parametric paired test).

    Use when normality assumption is violated for paired samples.

    Args:
        group1: First group of observations.
        group2: Second group of observations (paired with group1).
        alternative: One of 'two-sided', 'less', or 'greater'.

    Returns:
        Dictionary with statistic and p_value.
    """
    stat, p_value = stats.wilcoxon(group1, group2, alternative=alternative)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "n": len(group1),
    }


def normality_test(data: np.ndarray) -> Dict[str, float]:
    """Test for normality using Shapiro-Wilk test.

    Args:
        data: Array of observations (n >= 3).

    Returns:
        Dictionary with statistic and p_value.
        If p_value < 0.05, normality assumption is rejected.
    """
    if len(data) < 3:
        return {"statistic": np.nan, "p_value": np.nan, "is_normal": False}

    stat, p_value = stats.shapiro(data)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_normal": p_value >= 0.05,
    }


def format_mean_std(
    data: np.ndarray,
    precision: int = 2,
) -> str:
    """Format data as 'mean ± std' string.

    Args:
        data: Array of observations.
        precision: Decimal places for formatting.

    Returns:
        Formatted string like '0.85 ± 0.03'.
    """
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def format_mean_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    precision: int = 2,
) -> str:
    """Format data as 'mean [CI_low, CI_high]' string.

    Args:
        data: Array of observations.
        confidence: Confidence level.
        precision: Decimal places for formatting.

    Returns:
        Formatted string like '0.85 [0.82, 0.88]'.
    """
    mean, ci_low, ci_high = compute_confidence_interval(data, confidence)
    return f"{mean:.{precision}f} [{ci_low:.{precision}f}, {ci_high:.{precision}f}]"


def significance_marker(p_value: float) -> str:
    """Return significance marker based on p-value.

    Args:
        p_value: The p-value from a statistical test.

    Returns:
        '***' for p < 0.001, '**' for p < 0.01, '*' for p < 0.05, '' otherwise.
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


def aggregate_results(
    results: List[Dict[str, float]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate results from multiple runs into summary statistics.

    Args:
        results: List of result dictionaries (one per run/seed).
        metrics: List of metric names to aggregate. If None, uses all keys.

    Returns:
        Dictionary mapping metric names to {mean, std, se, ci_low, ci_high}.
    """
    if not results:
        return {}

    if metrics is None:
        metrics = list(results[0].keys())

    aggregated = {}
    for metric in metrics:
        values = np.array([r.get(metric, 0.0) for r in results])
        mean, ci_low, ci_high = compute_confidence_interval(values)
        aggregated[metric] = {
            "mean": mean,
            "std": float(np.std(values, ddof=1)),
            "se": compute_std_error(values),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": len(values),
        }
    return aggregated


def compare_methods(
    baseline_results: List[Dict[str, float]],
    method_results: List[Dict[str, float]],
    metrics: List[str],
    paired: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Compare two methods across multiple metrics with significance tests.

    Args:
        baseline_results: Results from baseline method (one dict per seed).
        method_results: Results from comparison method (one dict per seed).
        metrics: List of metric names to compare.
        paired: If True, use paired tests (same seeds). Otherwise independent.

    Returns:
        Dictionary with comparison results for each metric.
    """
    comparisons = {}

    for metric in metrics:
        baseline_values = np.array([r.get(metric, 0.0) for r in baseline_results])
        method_values = np.array([r.get(metric, 0.0) for r in method_results])

        # Test for normality
        norm_base = normality_test(baseline_values)
        norm_method = normality_test(method_values)
        use_parametric = norm_base["is_normal"] and norm_method["is_normal"]

        if paired:
            if use_parametric:
                test_result = paired_ttest(method_values, baseline_values)
                test_name = "paired_ttest"
            else:
                test_result = wilcoxon_test(method_values, baseline_values)
                test_name = "wilcoxon"
        else:
            if use_parametric:
                test_result = independent_ttest(method_values, baseline_values)
                test_name = "welch_ttest"
            else:
                test_result = mann_whitney_test(method_values, baseline_values)
                test_name = "mann_whitney"

        # Compute improvement
        baseline_mean = np.mean(baseline_values)
        method_mean = np.mean(method_values)
        if baseline_mean != 0:
            pct_change = (method_mean - baseline_mean) / abs(baseline_mean) * 100
        else:
            pct_change = 0.0

        comparisons[metric] = {
            "baseline_mean": float(baseline_mean),
            "baseline_std": float(np.std(baseline_values, ddof=1)),
            "method_mean": float(method_mean),
            "method_std": float(np.std(method_values, ddof=1)),
            "pct_change": float(pct_change),
            "test_name": test_name,
            "p_value": test_result["p_value"],
            "significant": test_result["p_value"] < 0.05,
            "marker": significance_marker(test_result["p_value"]),
        }

    return comparisons


def generate_latex_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str],
    method_names: List[str],
    caption: str = "Results comparison",
    label: str = "tab:results",
    precision: int = 3,
) -> str:
    """Generate a LaTeX table from aggregated results.

    Args:
        results: Nested dict: method_name -> metric -> {mean, std, ...}
        metrics: List of metric names (columns).
        method_names: List of method names (rows).
        caption: Table caption.
        label: LaTeX label for referencing.
        precision: Decimal places for values.

    Returns:
        LaTeX table string.
    """
    # Header
    cols = "l" + "r" * len(metrics)
    header = " & ".join(["Method"] + [m.replace("_", " ").title() for m in metrics])

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        f"{header} \\\\",
        "\\midrule",
    ]

    # Data rows
    for method in method_names:
        row_data = [method]
        for metric in metrics:
            if method in results and metric in results[method]:
                mean = results[method][metric]["mean"]
                std = results[method][metric]["std"]
                marker = results[method][metric].get("marker", "")
                row_data.append(f"${mean:.{precision}f} \\pm {std:.{precision}f}${marker}")
            else:
                row_data.append("--")
        lines.append(" & ".join(row_data) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: Array of observations.
        statistic: 'mean' or 'median'.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (estimate, ci_low, ci_high).
    """
    rng = np.random.default_rng(random_state)
    n = len(data)

    stat_func = np.mean if statistic == "mean" else np.median

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_high = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return float(stat_func(data)), float(ci_low), float(ci_high)
