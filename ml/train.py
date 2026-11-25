import os
import multiprocessing as mp

# Set joblib and loky environment options before importing ML libraries
os.environ["JOBLIB_START_METHOD"] = "spawn"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent thread contention between OpenMP & joblib

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from .artifacts.model_manager import ArtifactManager
from .config import DATASET_PATH, MODEL_OUTPUT_DIR
from .dataset.dataset_loader import AvalancheDataset
from .dataset.feature_engineering import FeatureEngineer, prune_avalanche_features, add_regional_risk_features
from .training.metrics import evaluate_binary, metrics_to_dict
from .training.splitter import TemporalSplitter
from .training.thresholds import ThresholdOptimizer
from .training.trainer import AvalancheTrainer


def run_training(
    dataset_path=DATASET_PATH,
    output_dir=MODEL_OUTPUT_DIR,
    targets=None,
):
    print("=" * 70)
    print("AVALANCHE ML TRAINING PIPELINE")
    print("=" * 70)

    dataset = AvalancheDataset(dataset_path)
    bundle = dataset.prepare()

    if targets is None:
        targets = bundle.forecast_targets

    missing = [t for t in targets if t not in bundle.df.columns]
    if missing:
        raise ValueError(f"Missing requested targets: {missing}")

    # =========================================================
    # 1. PRUNE REDUNDANT FEATURES
    # =========================================================
    pruned_df, dropped_cols = prune_avalanche_features(bundle.df, verbose=True)

    # =========================================================
    # 2. ADD REGIONAL RISK ENCODING
    # =========================================================
    pruned_df = add_regional_risk_features(pruned_df, target_cols=targets)

    # Re-build Feature Engineer with updated DataFrame
    feature_bundle = FeatureEngineer(pruned_df).build()
    X = feature_bundle.X

    print("\n===== FEATURE MATRIX =====")
    print(f"Rows: {len(X):,}")
    print(f"Features: {len(X.columns):,}")
    print(
        "Restricted features:",
        feature_bundle.restricted_feature_weights,
    )

    # =========================================================
    # ROLLING-ORIGIN VALIDATION
    # =========================================================
    temporal_splits = TemporalSplitter().rolling_split(
        pruned_df,
        season_column="season",
        train_seasons=3,
    )

    # =========================================================
    # FINAL TEST SPLIT
    # =========================================================
    final_train, final_test = TemporalSplitter().final_test_split(
        pruned_df,
        season_column="season",
    )

    trainer = AvalancheTrainer()
    # Change if needed
    threshold_optimizer = ThresholdOptimizer()
    artifact_manager = ArtifactManager(output_dir)

    rows = []

    for target in targets:

        print("\n" + "=" * 70)
        print(f"TRAINING TARGET: {target}")
        print("=" * 70)

        def target_diagnostics(df, target):
            y = df[target]
            pos_count = int(y.sum())
            total_count = len(y)
            pos_rate = float(y.mean())

            print(f"\n===== {target} =====")
            print(f"Total: {total_count}")
            print(f"Positive: {pos_count}")
            print(f"Negative: {total_count - pos_count}")
            print(f"Rate: {pos_rate:.4%}")

            if pos_count < 50:
                print(
                    f"⚠️ WARNING: Extremely low positive count ({pos_count}). "
                    "Ensure PR-AUC evaluation & pos_weight scaling are active."
                )

            print("\nBy mountain:")
            print(
                df.groupby("mountain")[target]
                .agg(["count", "sum", "mean"])
            )

            print("\nBy year:")
            print(
                df.groupby(df["date"].dt.year)[target]
                .agg(["count", "sum", "mean"])
            )

            print("\nBy month:")
            print(
                df.groupby(df["date"].dt.month)[target]
                .agg(["count", "sum", "mean"])
            )

        df = pruned_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        target_diagnostics(df, target)

        # =====================================================
        # ROLLING VALIDATION + FINAL MODEL
        # =====================================================
        result = trainer.train_target(
            target=target,
            feature_matrix=X,
            dataframe=df,
            temporal_splits=temporal_splits,
        )

        # =====================================================
        # THRESHOLD OPTIMIZATION
        # =====================================================
        y_valid = result.validation_targets
        valid_probabilities = result.validation_probabilities

        if y_valid is None or valid_probabilities is None:
            raise RuntimeError(
                f"Missing rolling validation predictions "
                f"for target '{target}'."
            )

        threshold = threshold_optimizer.optimize(
            y_valid,
            valid_probabilities,
        )

        # =====================================================
        # FINAL TEST EVALUATION
        # =====================================================
        test_idx = final_test.index.intersection(X.index)
        X_test = X.loc[test_idx]

        y_test = (
            df.loc[
                test_idx,
                target,
            ]
            .astype(int)
        )

        test_probabilities = result.model.predict_proba(X_test)[:, 1]

        # =====================================================
        # TEST METRICS
        # =====================================================
        test_metrics = evaluate_binary(
            target,
            y_test,
            test_probabilities,
            threshold.threshold,
        )

        test_dict = metrics_to_dict(test_metrics)

        # Save Artifacts
        result.threshold = threshold.threshold

        artifact_manager.save_target(
            result,
            threshold,
            test_dict,
        )

        # Summary Entry
        rows.append({
            "target": target,
            "validation_pr_auc": result.optimization.best_validation_pr_auc,
            "validation_roc_auc": result.optimization.best_validation_roc_auc,
            "seasonal_importance": result.optimization.seasonal_importance,
            "threshold": threshold.threshold,
            "validation_f1": threshold.f1,
            "test_pr_auc": test_metrics.pr_auc,
            "test_roc_auc": test_metrics.roc_auc,
            "test_f1": test_metrics.f1,
            "test_precision": test_metrics.precision,
            "test_recall": test_metrics.recall,
            "test_specificity": test_metrics.specificity,
            "test_balanced_accuracy": test_metrics.balanced_accuracy,
            "test_mcc": test_metrics.mcc,
        })

        print("\n" + "-" * 70)
        print(f"TEST RESULTS: {target}")
        print("-" * 70)
        print(f"PR-AUC:            {test_metrics.pr_auc:.5f}")
        print(f"ROC-AUC:           {test_metrics.roc_auc:.5f}")
        print(f"F1:                {test_metrics.f1:.5f}")
        print(f"Precision:         {test_metrics.precision:.5f}")
        print(f"Recall:            {test_metrics.recall:.5f}")
        print(f"Specificity:       {test_metrics.specificity:.5f}")
        print(f"Balanced accuracy: {test_metrics.balanced_accuracy:.5f}")
        print(f"MCC:               {test_metrics.mcc:.5f}")

    # =========================================================
    # SAVE SUMMARY
    # =========================================================
    summary = pd.DataFrame(rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "training_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Models: {output_dir}")
    print(f"Summary: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train optimized avalanche models."
    )
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--output", default=str(MODEL_OUTPUT_DIR))
    parser.add_argument(
        "--target", action="append",
        help="Specific target; repeat for multiple targets."
    )
    args = parser.parse_args()

    print("Targets Check:")
    print(args.target)

    run_training(
        dataset_path=args.dataset,
        output_dir=args.output,
        targets=args.target,
    )


if __name__ == "__main__":
    main()