"""
Report Generator — side-by-side comparison of Path A vs Path B vs Path C RAGAS scores.

Reads the saved RAGAS results for all three paths and produces:
    results/reports/comparison_report.csv   — per-question scores side by side
    results/reports/comparison_report.json  — same data as JSON
    results/reports/summary_report.json     — mean scores, deltas, winner per metric
    results/reports/stratified_report.csv   — scores grouped by layout_sensitivity / question_type
    results/reports/stratified_report.json  — same data as JSON

Expected inputs:
    results/path_a/ragas_results_path_a.json
    results/path_b/ragas_results_path_b.json
    results/path_c/ragas_results_path_c.json
"""

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load and return the central YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

_META_COLS = [
    "question_id", "layout_sensitivity", "question_type",
    "company_id", "sector", "doc_type",
]


def load_ragas_results(results_path: Path, path_label: str) -> pd.DataFrame:
    """Load a saved RAGAS results JSON into a DataFrame.

    Args:
        results_path: Directory containing ragas_results_<path_label>.json.
        path_label:   "path_a", "path_b", or "path_c".

    Returns:
        DataFrame with per-question scores.

    Raises:
        FileNotFoundError: If the results file does not exist.
    """
    json_path = results_path / f"ragas_results_{path_label}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {json_path}\n"
            f"Run the {path_label} pipeline first."
        )
    df = pd.read_json(json_path, orient="records")
    logger.info("Loaded %s results ← %s  (%d rows)", path_label, json_path, len(df))
    return df


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------

def build_comparison_df(
    dfs: dict[str, pd.DataFrame],
    metrics: list[str],
) -> pd.DataFrame:
    """Merge scores from multiple paths into a single side-by-side DataFrame.

    For each metric, columns are suffixed _<path_label>.
    Metadata columns from the first path are included once.

    Args:
        dfs:     Dict mapping path_label (e.g. "path_a") to its RAGAS DataFrame.
        metrics: List of metric names to include.

    Returns:
        DataFrame with columns: question_id, question, layout_sensitivity, …,
        <metric>_path_a, <metric>_path_b, <metric>_path_c, …
    """
    labels = list(dfs.keys())
    first_label = labels[0]
    df_first = dfs[first_label]

    available_metrics = [
        m for m in metrics
        if all(m in df for df in dfs.values())
    ]
    missing = set(metrics) - set(available_metrics)
    if missing:
        logger.warning("Metrics not found in all result sets, skipping: %s", missing)

    meta_present = [c for c in _META_COLS if c in df_first.columns]
    has_q = "question" in df_first.columns

    # Build base from first path (carries metadata)
    cols_first = meta_present + (["question"] if has_q else []) + available_metrics
    comparison = df_first[cols_first].copy()
    comparison.rename(columns={m: f"{m}_{first_label}" for m in available_metrics}, inplace=True)

    # Merge in subsequent paths
    for label in labels[1:]:
        df = dfs[label]
        q_col = ["question"] if "question" in df.columns else []
        sub = df[q_col + available_metrics].copy()
        sub.rename(columns={m: f"{m}_{label}" for m in available_metrics}, inplace=True)

        if "question" in comparison.columns and "question" in sub.columns:
            comparison = pd.merge(comparison, sub, on="question", how="outer")
        else:
            sub_no_q = sub.drop(columns=["question"], errors="ignore")
            comparison = pd.concat(
                [comparison.reset_index(drop=True), sub_no_q.reset_index(drop=True)], axis=1
            )

    return comparison


def build_summary(
    dfs: dict[str, pd.DataFrame],
    metrics: list[str],
) -> dict:
    """Compute mean scores and per-metric winners across all paths.

    Args:
        dfs:     Dict mapping path_label to its RAGAS DataFrame.
        metrics: List of metric names to summarise.

    Returns:
        Dict with structure:
        {
          "metrics": {
            "<metric_name>": {
              "path_a_mean": float, "path_b_mean": float, "path_c_mean": float,
              "winner": str,  # label of the highest-scoring path
              "means": {"path_a": float, "path_b": float, "path_c": float},
            },
            ...
          },
          "overall_winner": str,
          "wins": {"path_a": int, "path_b": int, "path_c": int},
          "ties": int,
        }
    """
    labels = list(dfs.keys())
    metric_summaries = {}
    wins = {label: 0 for label in labels}
    ties = 0

    for metric in metrics:
        missing_in = [l for l in labels if metric not in dfs[l].columns]
        if missing_in:
            logger.warning("Metric '%s' missing from %s — skipping.", metric, missing_in)
            continue

        means = {label: float(dfs[label][metric].dropna().mean()) for label in labels}

        max_mean = max(means.values())
        winners = [l for l in labels if abs(means[l] - max_mean) < 1e-4]

        if len(winners) == len(labels):
            winner = "tie"
            ties += 1
        elif len(winners) == 1:
            winner = winners[0]
            wins[winner] += 1
        else:
            winner = "+".join(winners)
            for w in winners:
                wins[w] += 0.5

        entry = {
            "means":  {l: round(means[l], 6) for l in labels},
            "winner": winner,
        }
        # Keep flat keys for backwards compat with old JSON readers
        for label in labels:
            entry[f"{label}_mean"] = round(means[label], 6)

        metric_summaries[metric] = entry

        logger.info(
            "  %-22s  " + "  ".join(f"{l.upper()}=%.4f" for l in labels) + "  winner=%s",
            metric,
            *[means[l] for l in labels],
            winner,
        )

    best_label = max(wins, key=lambda l: wins[l])
    # If tied for first, call it a tie
    top_wins = wins[best_label]
    if sum(1 for v in wins.values() if v == top_wins) > 1:
        overall_winner = "tie"
    else:
        overall_winner = best_label

    return {
        "metrics":        metric_summaries,
        "overall_winner": overall_winner,
        "wins":           {l: int(wins[l]) for l in labels},
        "ties":           ties,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_comparison_report(comparison_df: pd.DataFrame, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(reports_dir / "comparison_report.csv", index=False)
    logger.info("Saved comparison CSV  → %s", reports_dir / "comparison_report.csv")
    comparison_df.to_json(
        reports_dir / "comparison_report.json", orient="records", indent=2, force_ascii=False
    )
    logger.info("Saved comparison JSON → %s", reports_dir / "comparison_report.json")


def save_summary_report(summary: dict, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "summary_report.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("Saved summary JSON    → %s", json_path)


def print_summary_table(summary: dict) -> None:
    labels = list(next(iter(summary["metrics"].values()))["means"].keys())

    header = "  RAGAS COMPARISON SUMMARY — " + " vs ".join(l.upper().replace("_", " ") for l in labels)
    logger.info("")
    logger.info("=" * max(70, len(header) + 2))
    logger.info(header)
    logger.info("=" * max(70, len(header) + 2))

    col_fmt = "  {:<22}" + "  {:>8}" * len(labels) + "  {}"
    logger.info(col_fmt.format("Metric", *[l.upper()[-6:] for l in labels], "Winner"))
    logger.info("  " + "-" * (22 + 10 * len(labels) + 10))

    for metric, vals in summary["metrics"].items():
        means = vals["means"]
        logger.info(
            col_fmt.format(
                metric,
                *[f"{means[l]:.4f}" for l in labels],
                vals["winner"].upper(),
            )
        )

    logger.info("  " + "-" * (22 + 10 * len(labels) + 10))
    wins = summary["wins"]
    logger.info(
        "  Overall winner: %s  (%s)",
        summary["overall_winner"].upper(),
        "  ".join(f"{l.upper()[-6:]} wins: {wins[l]}" for l in labels),
    )
    logger.info("=" * max(70, len(header) + 2))
    logger.info("")


# ---------------------------------------------------------------------------
# Stratified analysis
# ---------------------------------------------------------------------------

def build_stratified_summary(
    comparison_df: pd.DataFrame,
    metrics: list[str],
    labels: list[str],
) -> dict:
    """Compute mean scores per path grouped by layout_sensitivity and question_type.

    Args:
        comparison_df: DataFrame from build_comparison_df() with metadata columns.
        metrics:       List of base metric names (without path suffix).
        labels:        List of path labels present in comparison_df.

    Returns:
        Nested dict:  {group_col: {group_value: {"n": int, "metrics": {...}}}}
    """
    result: dict = {}

    for group_col in ("layout_sensitivity", "question_type"):
        if group_col not in comparison_df.columns:
            logger.warning("Column '%s' not in comparison DataFrame — skipping stratification.", group_col)
            continue

        result[group_col] = {}
        for group_val, grp in comparison_df.groupby(group_col):
            metric_stats: dict = {}
            for metric in metrics:
                path_means = {}
                for label in labels:
                    col = f"{metric}_{label}"
                    if col in grp.columns and grp[col].notna().any():
                        path_means[label] = float(grp[col].dropna().mean())
                    else:
                        path_means[label] = None

                non_null = {l: v for l, v in path_means.items() if v is not None}
                if len(non_null) >= 2:
                    best_label = max(non_null, key=lambda l: non_null[l])
                    best_val = non_null[best_label]
                    tied = [l for l in non_null if abs(non_null[l] - best_val) < 1e-4]
                    winner = "tie" if len(tied) > 1 else best_label
                else:
                    winner = None

                metric_stats[metric] = {
                    "means":  {l: (round(v, 6) if v is not None else None) for l, v in path_means.items()},
                    "winner": winner,
                }
                # Flat keys for back-compat
                for label in labels:
                    metric_stats[metric][f"{label}_mean"] = metric_stats[metric]["means"][label]

            result[group_col][str(group_val)] = {"n": len(grp), "metrics": metric_stats}

    return result


def save_stratified_report(stratified: dict, reports_dir: Path, labels: list[str]) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "stratified_report.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(stratified, fh, indent=2, ensure_ascii=False)
    logger.info("Saved stratified JSON → %s", json_path)

    rows = []
    for group_col, groups in stratified.items():
        for group_val, info in groups.items():
            n = info["n"]
            for metric, vals in info["metrics"].items():
                row = {
                    "group_by":    group_col,
                    "group_value": group_val,
                    "n":           n,
                    "metric":      metric,
                }
                for label in labels:
                    row[f"{label}_mean"] = vals["means"].get(label)
                row["winner"] = vals.get("winner", "")
                rows.append(row)

    if rows:
        flat_df = pd.DataFrame(rows)
        csv_path = reports_dir / "stratified_report.csv"
        flat_df.to_csv(csv_path, index=False)
        logger.info("Saved stratified CSV  → %s", csv_path)


def print_stratified_table(stratified: dict, labels: list[str]) -> None:
    for group_col, groups in stratified.items():
        logger.info("")
        logger.info("  Stratified by: %s", group_col.upper())
        header = "  {:<10}  {:>3}  {:<22}" + "  {:>8}" * len(labels) + "  {}"
        logger.info(header.format("Group", "N", "Metric", *[l.upper()[-6:] for l in labels], "Winner"))
        logger.info("  " + "-" * (10 + 5 + 22 + 10 * len(labels) + 10))
        for group_val, info in sorted(groups.items()):
            n = info["n"]
            for metric, vals in info["metrics"].items():
                means = vals["means"]
                row_vals = [
                    f"{means[l]:.4f}" if means.get(l) is not None else "  N/A  "
                    for l in labels
                ]
                logger.info(
                    header.format(group_val, n, metric, *row_vals, (vals.get("winner") or "").upper())
                )


# ---------------------------------------------------------------------------
# High-level runner (called by run_pipeline.py)
# ---------------------------------------------------------------------------

def generate_report(config_path: Path | None = None) -> dict:
    """Generate the full comparison report from saved RAGAS result files.

    Loads results for all three paths (A, B, C), builds the side-by-side
    comparison DataFrame and summary dict, saves all output files, and
    prints the summary table.  Path C results are optional — if the file is
    missing the report falls back to A-vs-B only.

    Args:
        config_path: Optional config file path override.

    Returns:
        Summary dict from build_summary().
    """
    project_root = Path(__file__).resolve().parents[2]
    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"

    cfg = load_config(config_path)

    log_cfg = cfg.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s — %(message)s"),
        datefmt=log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"),
    )

    reports_dir = project_root / cfg["paths"]["reports"]
    metrics     = cfg["evaluation"]["metrics"]

    # Load each path; path C is optional
    path_dirs = {
        "path_a": project_root / cfg["paths"]["results_path_a"],
        "path_b": project_root / cfg["paths"]["results_path_b"],
        "path_c": project_root / cfg["paths"]["results_path_c"],
    }

    dfs: dict[str, pd.DataFrame] = {}
    for label, results_dir in path_dirs.items():
        try:
            dfs[label] = load_ragas_results(results_dir, label)
        except FileNotFoundError as exc:
            if label == "path_c":
                logger.warning("Path C results not found — reporting A vs B only. (%s)", exc)
            else:
                logger.error("%s", exc)
                return {}

    if len(dfs) < 2:
        logger.error("Need at least two paths with RAGAS results to generate a report.")
        return {}

    labels = list(dfs.keys())
    logger.info("=== Generating comparison report for: %s ===", ", ".join(labels))

    comparison_df = build_comparison_df(dfs, metrics)
    save_comparison_report(comparison_df, reports_dir)

    summary = build_summary(dfs, metrics)
    save_summary_report(summary, reports_dir)
    print_summary_table(summary)

    stratified = build_stratified_summary(comparison_df, metrics, labels)
    save_stratified_report(stratified, reports_dir, labels)
    print_stratified_table(stratified, labels)

    logger.info("=== Report generation complete ===")
    return summary
