import joblib

class AvalancheModelWrapper:
    def __init__(self, model, target_name: str, threshold: float):
        self.model = model
        self.target_name = target_name
        self.threshold = threshold

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def save(self, filepath: str):
        joblib.dump({
            "model": self.model,
            "target_name": self.target_name,
            "threshold": self.threshold
        }, filepath)

    @classmethod
    def load(cls, filepath: str):
        data = joblib.load(filepath)
        return cls(data["model"], data["target_name"], data["threshold"])