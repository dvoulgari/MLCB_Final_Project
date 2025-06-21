from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PipelineBuilder:
    def __init__(self):
        pass  # no model in constructor

    def build(self, model):
        if model is None:
            raise ValueError("You must provide a model to build the pipeline.")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
