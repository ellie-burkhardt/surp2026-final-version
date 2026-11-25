"""
research_plots.py

Generate publication/poster figures from the avalanche ML dataset and
training_summary.csv.

Season definition used for all seasonal/date-based avalanche analyses:
    October 1 through April 30.

Season labels are named by the October start year, e.g.:
    2024-10 through 2025-04 -> 2024-2025

Expected project structure:
    data/processed/final_avalanche_dataset.parquet
    trained_models/training_summary.csv

Outputs:
    research_outputs/figures/*.png
    research_outputs/figures/*.pdf
    research_outputs/seasonal_avalanche_prevalence.csv
    research_outputs/seasonal_avalanche_counts.csv
    research_outputs/poster_metrics.csv

Run from the project root:
    python research_plots.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "final_avalanche_dataset.parquet"
SUMMARY_PATH = PROJECT_ROOT / "trained_models" / "training_summary.csv"

OUTPUT_DIR = PROJECT_ROOT / "research_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

PROBLEMS = ["wind_slab", "storm_slab", "wet_loose", "persistent_slab"]

DISPLAY_NAMES = {
    "wind_slab": "Wind Slab",
    "storm_slab": "Storm Slab",
    "wet_loose": "Wet Loose",
    "persistent_slab": "Persistent Slab",
}

SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4]

# Poster/research-friendly plotting settings.
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.autolayout": True,
})


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def save_figure(fig, name):
    """Save each figure as high-resolution PNG and vector PDF."""
    png_path = FIGURE_DIR / f"{name}.png"
    pdf_path = FIGURE_DIR / f"{name}.pdf"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ {png_path.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ {pdf_path.relative_to(PROJECT_ROOT)}")


def find_column(df, candidates):
    """Find the first matching column from a list of possible names."""
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def normalize_problem_name(value):
    """Convert common target names to the four canonical problem names."""
    text = str(value).lower()

    for problem in PROBLEMS:
        if problem in text:
            return problem

    return None


def target_column(df, problem):
    """Find the best available binary target column for a problem."""
    return find_column(
        df,
        [
            f"{problem}_present_future",
            f"{problem}_future",
            f"{problem}_present",
        ],
    )


def add_season_columns(df, date_col):
    """
    Add season_start_year and season to a copy of df.

    Avalanche season is October-April:
        Oct-Dec 2024 -> 2024-2025
        Jan-Apr 2025 -> 2024-2025

    May-September receive NaN season values and are excluded from seasonal
    avalanche analyses.
    """
    temp = df.copy()
    dates = pd.to_datetime(temp[date_col], errors="coerce")
    month = dates.dt.month
    year = dates.dt.year

    valid = month.isin(SEASON_MONTHS)
    season_start = np.where(
        month >= 10,
        year,
        year - 1,
    )

    temp["season_start_year"] = pd.Series(season_start, index=temp.index).where(valid)
    temp["season"] = np.nan

    valid_start = temp["season_start_year"].notna()
    start_year = temp.loc[valid_start, "season_start_year"].astype(int)
    temp.loc[valid_start, "season"] = (
        start_year.astype(str)
        + "-"
        + (start_year + 1).astype(str)
    )

    return temp


def seasonal_dataframe(df):
    """Return rows whose dates fall in the October-April avalanche season."""
    date_col = find_column(df, ["date"])
    if date_col is None:
        return pd.DataFrame()

    temp = add_season_columns(df, date_col)
    temp = temp[temp["season"].notna()].copy()
    temp["season_start_year"] = temp["season_start_year"].astype(int)
    return temp.sort_values(["season_start_year", date_col])


def season_order(df):
    """Return chronologically ordered season labels."""
    if df.empty or "season_start_year" not in df.columns:
        return []

    return (
        df[["season", "season_start_year"]]
        .drop_duplicates()
        .sort_values("season_start_year")["season"]
        .tolist()
    )


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_dataset():
    print("\nLoading dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found:\n{DATASET_PATH}")

    df = pd.read_parquet(DATASET_PATH)

    print(f"  ✓ Rows: {len(df):,}")
    print(f"  ✓ Columns: {len(df):,}")

    date_col = find_column(df, ["date"])
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    return df


def load_training_summary():
    print("\nLoading training summary...")

    if not SUMMARY_PATH.exists():
        print(
            f"  ⚠ training_summary.csv not found:\n"
            f"    {SUMMARY_PATH}\n"
            f"  Performance figures will be skipped."
        )
        return None

    summary = pd.read_csv(SUMMARY_PATH)

    print(f"  ✓ Rows: {len(summary):,}")
    print(f"  ✓ Columns: {len(summary.columns):,}")
    print(f"  ✓ Columns: {list(summary.columns)}")

    return summary


# ---------------------------------------------------------------------
# Extract model performance
# ---------------------------------------------------------------------

def extract_performance(summary):
    """Extract model metrics from training_summary.csv."""
    if summary is None or summary.empty:
        return pd.DataFrame()

    records = []
    problem_col = None

    for col in summary.columns:
        values = summary[col].astype(str).str.lower()
        if values.str.contains(
            "wind_slab|storm_slab|wet_loose|persistent_slab",
            regex=True,
        ).any():
            problem_col = col
            break

    if problem_col is None:
        print("  ⚠ Could not identify the target/problem column.")
        return pd.DataFrame()

    column_map = {}

    for col in summary.columns:
        low = str(col).lower().replace("-", "_").replace(" ", "_")

        if "pr_auc" in low or "prauc" in low:
            # Prefer test PR-AUC over validation PR-AUC if possible.
            if "test" in low:
                column_map["test_pr_auc"] = col
            elif "mean" in low or "validation" in low:
                column_map.setdefault("mean_pr_auc", col)
            else:
                column_map.setdefault("pr_auc", col)

        elif "roc_auc" in low or "rocauc" in low:
            if "test" in low:
                column_map["test_roc_auc"] = col
            else:
                column_map.setdefault("roc_auc", col)

        elif low == "f1" or low.endswith("_f1"):
            if "test" in low:
                column_map["test_f1"] = col
            else:
                column_map.setdefault("f1", col)

        elif "precision" in low:
            column_map.setdefault("test_precision", col)

        elif "recall" in low or "sensitivity" in low:
            column_map.setdefault("test_recall", col)

        elif "specificity" in low:
            column_map.setdefault("test_specificity", col)

        elif "balanced_accuracy" in low:
            column_map.setdefault("test_balanced_accuracy", col)

        elif low == "mcc" or low.endswith("_mcc"):
            column_map.setdefault("test_mcc", col)

        elif "fold1_pr_auc" in low or "fold_1_pr_auc" in low:
            column_map["fold1_pr_auc"] = col

        elif "fold2_pr_auc" in low or "fold_2_pr_auc" in low:
            column_map["fold2_pr_auc"] = col

    # Explicitly map common current schema names.
    explicit = {
        "validation_pr_auc": "validation_pr_auc",
        "validation_roc_auc": "validation_roc_auc",
        "test_pr_auc": "test_pr_auc",
        "test_roc_auc": "test_roc_auc",
        "test_f1": "test_f1",
        "test_precision": "test_precision",
        "test_recall": "test_recall",
        "test_specificity": "test_specificity",
        "test_balanced_accuracy": "test_balanced_accuracy",
        "test_mcc": "test_mcc",
    }

    for destination, source in explicit.items():
        if source in summary.columns:
            column_map[destination] = source

    print(f"  ✓ Detected problem column: {problem_col}")
    print(f"  ✓ Detected metrics: {column_map}")

    for _, row in summary.iterrows():
        problem = normalize_problem_name(row[problem_col])
        if problem is None:
            continue

        record = {
            "problem": problem,
            "problem_display": DISPLAY_NAMES[problem],
        }

        for metric, col in column_map.items():
            try:
                record[metric] = float(row[col])
            except (TypeError, ValueError):
                record[metric] = np.nan

        records.append(record)

    return pd.DataFrame(records)


def extract_metrics_from_training_log_fallback():
    """Fallback metrics from the supplied current training run."""
    return pd.DataFrame([
        {
            "problem": "wind_slab", "problem_display": "Wind slab",
            "test_pr_auc": 0.55714, "test_roc_auc": 0.81541,
            "test_f1": 0.62411, "test_precision": 0.53441,
            "test_recall": 0.75000, "test_specificity": 0.74501,
            "test_balanced_accuracy": 0.74751, "test_mcc": 0.45522,
            "fold1_pr_auc": 0.76173, "fold2_pr_auc": 0.61586,
            "mean_pr_auc": 0.68880,
        },
        {
            "problem": "storm_slab", "problem_display": "Storm slab",
            "test_pr_auc": 0.13458, "test_roc_auc": 0.71508,
            "test_f1": 0.02985, "test_precision": 0.05000,
            "test_recall": 0.02128, "test_specificity": 0.96724,
            "test_balanced_accuracy": 0.49426, "test_mcc": -0.01721,
            "fold1_pr_auc": 0.37995, "fold2_pr_auc": 0.36463,
            "mean_pr_auc": 0.37229,
        },
        {
            "problem": "wet_loose", "problem_display": "Wet loose",
            "test_pr_auc": 0.46483, "test_roc_auc": 0.74700,
            "test_f1": 0.46729, "test_precision": 0.51020,
            "test_recall": 0.43103, "test_specificity": 0.84106,
            "test_balanced_accuracy": 0.63605, "test_mcc": 0.28758,
            "fold1_pr_auc": 0.58534, "fold2_pr_auc": 0.43293,
            "mean_pr_auc": 0.50913,
        },
        {
            "problem": "persistent_slab", "problem_display": "Persistent Slab",
            "test_pr_auc": 0.30438, "test_roc_auc": 0.88386,
            "test_f1": 0.38596, "test_precision": 0.29730,
            "test_recall": 0.55000, "test_specificity": 0.95717,
            "test_balanced_accuracy": 0.75358, "test_mcc": 0.37821,
            "fold1_pr_auc": 0.77186, "fold2_pr_auc": 0.56646,
            "mean_pr_auc": 0.66916,
        },

    ])


# ---------------------------------------------------------------------
# Figure 1: Overall problem prevalence
# ---------------------------------------------------------------------

def plot_problem_prevalence(df):
    print("\n[1/11] Overall avalanche-problem prevalence")

    rates = []
    for problem in PROBLEMS:
        col = target_column(df, problem)
        if col is None:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        rates.append({
            "problem": DISPLAY_NAMES[problem],
            "rate": values.mean() * 100,
        })

    if not rates:
        print("  ⚠ No target columns found.")
        return

    data = pd.DataFrame(rates)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(data["problem"], data["rate"])

    ax.set_ylabel("Positive observations (%)")
    ax.set_xlabel("Avalanche problem")
    ax.set_title("Overall avalanche-problem prevalence")
    ax.set_ylim(0, max(40, data["rate"].max() * 1.2))

    for bar, value in zip(bars, data["rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "01_problem_prevalence")


# ---------------------------------------------------------------------
# Figure 2: Seasonal prevalence — PRIMARY requested graph
# ---------------------------------------------------------------------

def calculate_seasonal_prevalence(df):
    temp = seasonal_dataframe(df)
    if temp.empty:
        return pd.DataFrame()

    rows = []
    for problem in PROBLEMS:
        col = target_column(temp, problem)
        if col is None:
            continue

        grouped = (
            temp.groupby(["season_start_year", "season"])[col]
            .mean()
            .mul(100)
            .reset_index(name="prevalence_percent")
        )
        grouped["problem"] = DISPLAY_NAMES[problem]
        grouped["positive_count"] = (
            temp.groupby(["season_start_year", "season"])[col]
            .sum()
            .values
        )
        grouped["observation_count"] = (
            temp.groupby(["season_start_year", "season"])[col]
            .count()
            .values
        )
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True).sort_values(
        ["season_start_year", "problem"]
    )


def plot_seasonal_prevalence(df):
    print("\n[2/11] Avalanche-problem prevalence by season")

    seasonal = calculate_seasonal_prevalence(df)
    if seasonal.empty:
        print("  ⚠ No seasonal prevalence data.")
        return

    pivot = seasonal.pivot(
        index="season_start_year",
        columns="problem",
        values="prevalence_percent",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(marker="o", linewidth=2, ax=ax)

    ax.set_xticks(pivot.index)
    ax.set_xticklabels(
        [f"{y}-{y + 1}" for y in pivot.index],
        rotation=30,
        ha="right",
    )
    ax.set_xlabel("Avalanche season (October-April)")
    ax.set_ylabel("Prevalence (%)")
    ax.set_title("Avalanche-problem prevalence by season")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Avalanche problem", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "02_seasonal_avalanche_problem_prevalence")

    # Save the exact data behind the figure for research use.
    seasonal.to_csv(
        OUTPUT_DIR / "seasonal_avalanche_prevalence.csv",
        index=False,
    )
    print(f"  ✓ {OUTPUT_DIR / 'seasonal_avalanche_prevalence.csv'}")


# ---------------------------------------------------------------------
# Figure 3: Held-out PR-AUC
# ---------------------------------------------------------------------

def plot_test_prauc(performance):
    print("\n[3/11] Held-out test PR-AUC")

    if performance.empty or "test_pr_auc" not in performance:
        print("  ⚠ No test PR-AUC data.")
        return

    data = performance.copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(data["problem_display"], data["test_pr_auc"])

    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("Avalanche problem")
    ax.set_title("Held-out test performance: PR-AUC")
    ax.set_ylim(0, 1)

    for bar, value in zip(bars, data["test_pr_auc"]):
        if pd.notna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "03_test_prauc")


# ---------------------------------------------------------------------
# Figure 4: Held-out ROC-AUC
# ---------------------------------------------------------------------

def plot_test_rocauc(performance):
    print("\n[4/11] Held-out test ROC-AUC")

    if performance.empty or "test_roc_auc" not in performance:
        print("  ⚠ No test ROC-AUC data.")
        return

    data = performance.copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(data["problem_display"], data["test_roc_auc"])

    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("Avalanche problem")
    ax.set_title("Held-out test performance: ROC-AUC")
    ax.set_ylim(0, 1)

    for bar, value in zip(bars, data["test_roc_auc"]):
        if pd.notna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "04_test_rocauc")


# ---------------------------------------------------------------------
# Figure 5: PR-AUC vs ROC-AUC
# ---------------------------------------------------------------------

def plot_test_prauc_vs_rocauc(performance):
    print("\n[5/11] PR-AUC vs ROC-AUC")

    required = {"test_pr_auc", "test_roc_auc"}
    if performance.empty or not required.issubset(performance.columns):
        print("  ⚠ Required metrics unavailable.")
        return

    data = performance.copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(data))
    width = 0.36

    ax.bar(x - width / 2, data["test_pr_auc"], width, label="PR-AUC")
    ax.bar(x + width / 2, data["test_roc_auc"], width, label="ROC-AUC")

    ax.set_xticks(x)
    ax.set_xticklabels(data["problem_display"])
    ax.set_ylabel("Score")
    ax.set_xlabel("Avalanche problem")
    ax.set_title("Held-out test performance: PR-AUC vs ROC-AUC")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "05_test_prauc_vs_rocauc")


# ---------------------------------------------------------------------
# Figure 6: Classification metrics
# ---------------------------------------------------------------------

def plot_classification_metrics(performance):
    print("\n[6/11] Classification metrics")

    metrics = [
        ("test_f1", "F1"),
        ("test_precision", "Precision"),
        ("test_recall", "Recall"),
        ("test_specificity", "Specificity"),
        ("test_balanced_accuracy", "Balanced accuracy"),
    ]

    available = [m for m in metrics if m[0] in performance.columns]
    if not available:
        print("  ⚠ No classification metrics found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(performance))
    width = 0.15

    for i, (column, label) in enumerate(available):
        ax.bar(
            x + (i - (len(available) - 1) / 2) * width,
            performance[column],
            width,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(performance["problem_display"])
    ax.set_ylabel("Score")
    ax.set_xlabel("Avalanche problem")
    ax.set_title("Held-out test classification metrics")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "06_test_classification_metrics")


# ---------------------------------------------------------------------
# Figure 7: Rolling validation
# ---------------------------------------------------------------------

def plot_rolling_validation(performance):
    print("\n[7/11] Rolling validation PR-AUC")

    required = {"fold1_pr_auc", "fold2_pr_auc"}
    if performance.empty or not required.issubset(performance.columns):
        print("  ⚠ Rolling fold metrics unavailable.")
        return

    data = performance.copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(data))
    width = 0.35

    ax.bar(
        x - width / 2,
        data["fold1_pr_auc"],
        width,
        label="Fold 1: 2021–2023 → 2024",
    )
    ax.bar(
        x + width / 2,
        data["fold2_pr_auc"],
        width,
        label="Fold 2: 2021–2024 → 2025",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(data["problem_display"])
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("Avalanche problem")
    ax.set_title("Rolling-origin validation performance")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "07_rolling_validation_prauc")


# ---------------------------------------------------------------------
# Figure 8: Mountain prevalence
# ---------------------------------------------------------------------

def plot_mountain_prevalence(df):
    print("\n[8/11] Mountain-specific prevalence")

    mountain_col = find_column(df, ["mountain"])
    if mountain_col is None:
        print("  ⚠ Mountain column not found.")
        return

    rows = []
    for problem in PROBLEMS:
        col = target_column(df, problem)
        if col is None:
            continue

        grouped = df.groupby(mountain_col)[col].mean().mul(100)
        for mountain, rate in grouped.items():
            rows.append({
                "problem": DISPLAY_NAMES[problem],
                "mountain": str(mountain).title(),
                "rate": rate,
            })

    if not rows:
        print("  ⚠ No mountain prevalence data.")
        return

    data = pd.DataFrame(rows)
    pivot = data.pivot(index="mountain", columns="problem", values="rate")

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)

    ax.set_ylabel("Positive observations (%)")
    ax.set_xlabel("Mountain")
    ax.set_title("Avalanche-problem prevalence by mountain")
    ax.legend(title="Avalanche problem", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "08_mountain_prevalence")


# ---------------------------------------------------------------------
# Figure 9: Seasonal problem counts — useful companion to prevalence
# ---------------------------------------------------------------------

def plot_seasonal_problem_counts(df):
    print("\n[9/11] Avalanche-problem counts by season")

    temp = seasonal_dataframe(df)
    if temp.empty:
        print("  ⚠ No seasonal data.")
        return

    rows = []
    for problem in PROBLEMS:
        col = target_column(temp, problem)
        if col is None:
            continue

        grouped = (
            temp.groupby(["season_start_year", "season"])[col]
            .sum()
            .reset_index(name="positive_count")
        )
        grouped["problem"] = DISPLAY_NAMES[problem]
        rows.append(grouped)

    if not rows:
        return

    data = pd.concat(rows, ignore_index=True)
    data.to_csv(OUTPUT_DIR / "seasonal_avalanche_counts.csv", index=False)
    print(f"  ✓ {OUTPUT_DIR / 'seasonal_avalanche_counts.csv'}")

    pivot = data.pivot(
        index="season_start_year",
        columns="problem",
        values="positive_count",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax)

    ax.set_xlabel("Avalanche season (October-April)")
    ax.set_ylabel("Positive observations (count)")
    ax.set_title("Avalanche-problem occurrences by season")
    ax.set_xticklabels(
        [f"{y}-{y + 1}" for y in pivot.index],
        rotation=30,
        ha="right",
    )
    ax.legend(title="Avalanche problem", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "09_seasonal_problem_counts")


# ---------------------------------------------------------------------
# Figure 10: Average number of avalanche problems per observation/season
# ---------------------------------------------------------------------

def plot_seasonal_problem_burden(df):
    print("\n[10/11] Average avalanche-problem burden by season")

    temp = seasonal_dataframe(df)
    if temp.empty:
        print("  ⚠ No seasonal data.")
        return

    available = []
    for problem in PROBLEMS:
        col = target_column(temp, problem)
        if col is not None:
            available.append(col)

    if not available:
        print("  ⚠ No avalanche problem target columns found.")
        return

    numeric = temp[available].apply(pd.to_numeric, errors="coerce")
    temp = temp.copy()
    temp["problem_count"] = numeric.sum(axis=1, min_count=1)

    burden = (
        temp.groupby(["season_start_year", "season"])["problem_count"]
        .mean()
        .reset_index()
        .sort_values("season_start_year")
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(
        burden["season"],
        burden["problem_count"],
    )

    ax.set_xlabel("Avalanche season (October-April)")
    ax.set_ylabel("Mean number of problems per observation")
    ax.set_title("Average avalanche-problem burden by season")
    ax.tick_params(axis="x", rotation=30)

    for bar, value in zip(bars, burden["problem_count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, max(1, burden["problem_count"].max() * 1.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "10_seasonal_problem_burden")


# ---------------------------------------------------------------------
# Figure 11: Prevalence vs test PR-AUC
# ---------------------------------------------------------------------

def plot_prevalence_vs_prauc(df, performance):
    print("\n[11/11] Prevalence vs test PR-AUC")

    if performance.empty or "test_pr_auc" not in performance.columns:
        print("  ⚠ Test PR-AUC unavailable.")
        return

    rows = []
    for problem in PROBLEMS:
        col = target_column(df, problem)
        if col is None:
            continue

        prevalence = pd.to_numeric(df[col], errors="coerce").mean() * 100
        match = performance[performance["problem"] == problem]
        if match.empty:
            continue

        rows.append({
            "problem": DISPLAY_NAMES[problem],
            "prevalence": prevalence,
            "pr_auc": match.iloc[0]["test_pr_auc"],
        })

    data = pd.DataFrame(rows)
    if data.empty:
        print("  ⚠ No matching prevalence/performance data.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data["prevalence"], data["pr_auc"], s=80)

    for _, row in data.iterrows():
        ax.annotate(
            row["problem"],
            (row["prevalence"], row["pr_auc"]),
            xytext=(7, 7),
            textcoords="offset points",
        )

    ax.set_xlabel("Positive observations (%)")
    ax.set_ylabel("Held-out test PR-AUC")
    ax.set_title("Class prevalence vs model performance")
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "11_prevalence_vs_prauc")


# ---------------------------------------------------------------------
# Poster metrics CSV
# ---------------------------------------------------------------------

def create_poster_metrics(df, performance):
    print("\nCreating poster metrics CSV...")

    rows = []
    for problem in PROBLEMS:
        record = {"problem": DISPLAY_NAMES[problem]}

        target_col = target_column(df, problem)
        if target_col is not None:
            values = pd.to_numeric(df[target_col], errors="coerce")
            record["positive_count"] = int(values.sum())
            record["total_count"] = int(values.notna().sum())
            record["positive_rate_percent"] = values.mean() * 100

        if not performance.empty:
            match = performance[performance["problem"] == problem]
            if not match.empty:
                row = match.iloc[0]
                for source, destination in [
                    ("test_pr_auc", "test_pr_auc"),
                    ("test_roc_auc", "test_roc_auc"),
                    ("test_f1", "test_f1"),
                    ("test_precision", "test_precision"),
                    ("test_recall", "test_recall"),
                    ("test_specificity", "test_specificity"),
                    ("test_balanced_accuracy", "test_balanced_accuracy"),
                    ("test_mcc", "test_mcc"),
                    ("fold1_pr_auc", "fold1_pr_auc"),
                    ("fold2_pr_auc", "fold2_pr_auc"),
                    ("mean_pr_auc", "mean_validation_pr_auc"),
                    ("validation_pr_auc", "validation_pr_auc"),
                ]:
                    if source in row.index:
                        record[destination] = row[source]

        rows.append(record)

    output = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "poster_metrics.csv"
    output.to_csv(output_path, index=False)
    print(f"  ✓ {output_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print("AVALANCHE ML RESEARCH FIGURE GENERATOR")
    print("=" * 70)
    print("Season definition: October-April")
    print("Example: October 2024 through April 2025 = 2024-2025")
    print("=" * 70)

    df = load_dataset()
    summary = load_training_summary()

    performance = extract_performance(summary)
    if performance.empty:
        print("\nUsing current training-run metrics as fallback...")
        performance = extract_metrics_from_training_log_fallback()

    print("\nGenerating figures...")

    plot_problem_prevalence(df)
    plot_seasonal_prevalence(df)
    plot_test_prauc(performance)
    plot_test_rocauc(performance)
    plot_test_prauc_vs_rocauc(performance)
    plot_classification_metrics(performance)
    plot_rolling_validation(performance)
    plot_mountain_prevalence(df)
    plot_seasonal_problem_counts(df)
    plot_seasonal_problem_burden(df)
    plot_prevalence_vs_prauc(df, performance)

    create_poster_metrics(df, performance)

    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)
    print(f"Figures: {FIGURE_DIR}")
    print(f"Metrics: {OUTPUT_DIR / 'poster_metrics.csv'}")
    print(f"Seasonal prevalence: {OUTPUT_DIR / 'seasonal_avalanche_prevalence.csv'}")
    print(f"Seasonal counts: {OUTPUT_DIR / 'seasonal_avalanche_counts.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()