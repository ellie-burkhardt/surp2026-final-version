"""Load trained avalanche models and metadata."""
from pathlib import Path
import json
from catboost import CatBoostClassifier


class ModelLoader:
    """Loader for the four current avalanche-problem models."""

    PROBLEM_NAMES = [
        "storm_slab_present_future",
        "wind_slab_present_future",
        "wet_loose_present_future",
        "persistent_slab_present_future",
    ]

    def __init__(self, artifact_directory):
        self.artifact_directory = Path(artifact_directory)
        self.problem_names = list(self.PROBLEM_NAMES)

    def load_models(self):
        models = {}
        for problem in self.problem_names:
            path = self.artifact_directory / problem / "model.cbm"
            if not path.exists():
                raise FileNotFoundError(
                    f"Could not find model for '{problem}':\n{path.resolve()}"
                )
            model = CatBoostClassifier()
            model.load_model(str(path))
            models[problem] = model
        return models

    def load_metadata(self):
        metadata = {}
        for problem in self.problem_names:
            directory_metadata = self.artifact_directory / problem / "metadata.json"
            legacy_metadata = self.artifact_directory / f"{problem}_metadata.json"
            if directory_metadata.exists():
                path = directory_metadata
            elif legacy_metadata.exists():
                path = legacy_metadata
            else:
                raise FileNotFoundError(
                    f"Could not find metadata for '{problem}'. Checked:\n"
                    f"  {directory_metadata}\n  {legacy_metadata}"
                )
            with open(path, "r", encoding="utf-8") as f:
                metadata[problem] = json.load(f)
        return metadata

    def load_all(self):
        return {"models": self.load_models(), "metadata": self.load_metadata()}