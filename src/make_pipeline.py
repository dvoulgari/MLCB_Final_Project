from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


def make_pipeline(model=None, random_state=42):
    if model is None:
        model = LGBMClassifier(random_state=random_state)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
