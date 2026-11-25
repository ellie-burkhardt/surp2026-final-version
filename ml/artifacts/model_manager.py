from pathlib import Path
import json
import numpy as np

class ArtifactManager:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_target(self, target_result, threshold_result, test_metrics):
        target_dir = self.output_dir / target_result.target
        target_dir.mkdir(parents=True, exist_ok=True)
        model_path = target_dir / "model.cbm"
        metadata_path = target_dir / "metadata.json"

        target_result.model.save_model(str(model_path))

        metadata = {
            "target": target_result.target,
            "feature_names": target_result.feature_names,
            "threshold": threshold_result.threshold,
            "validation_threshold_f1": threshold_result.f1,
            "validation_threshold_precision": threshold_result.precision,
            "validation_threshold_recall": threshold_result.recall,
            "best_params": target_result.optimization.best_params,
            "best_iteration": target_result.optimization.best_iteration,
            "validation_pr_auc": target_result.optimization.best_validation_pr_auc,
            "validation_roc_auc": target_result.optimization.best_validation_roc_auc,
            "seasonal_importance": target_result.optimization.seasonal_importance,
            "seasonal_importances": target_result.optimization.seasonal_importances,
            "train_rows": target_result.train_rows,
            "validation_rows": target_result.validation_rows,
            "test_rows": target_result.test_rows,
            "train_positive_rate": target_result.train_positive_rate,
            "validation_positive_rate": target_result.validation_positive_rate,
            "test_positive_rate": target_result.test_positive_rate,
            "test_metrics": test_metrics,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=_json_default)
        )
        return model_path, metadata_path

def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")