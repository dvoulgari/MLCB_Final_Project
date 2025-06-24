import warnings
import optuna
from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from LightGBMBaselineClassifier import (
    LightGBMBaselineClassifier,
    DEFAULT_TEST_SIZE,
    DEFAULT_SEED,
    TRAINING_DATASET
)

# Suppress warnings to keep notebooks clean. Comment this out while debugging.
warnings.filterwarnings('ignore')

# Suppress Optuna logging to avoid cluttering the output.
# optuna.logging.set_verbosity(optuna.logging.CRITICAL)


class LightGBMTuner(LightGBMBaselineClassifier):
    def __init__(self, csv_path=TRAINING_DATASET, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_SEED, n_trials=50):
        super().__init__(csv_path, test_size, random_state)
        self.n_trials = n_trials
        self.best_params = None

    def tune(self):
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.random_state))
        study.optimize(self._objective, n_trials=self.n_trials)
        self.best_params = study.best_params

    def train_best_model(self):
        if self.best_params is None:
            raise ValueError("You must run tune() first.")

        tuned_model = LGBMClassifier(
            **self.best_params, random_state=self.random_state)
        self.pipeline = self.pipeline_builder.build(tuned_model)
        self.pipeline.fit(self.X_train, self.y_train)

    def print_best_params(self):
        print("Best params:", self.best_params)

    def _objective(self, trial):
        # Split training data into train/validation (Optuna inner loop)
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train,
            self.y_train,
            test_size=self.test_size,
            stratify=self.y_train,
            random_state=self.random_state
        )

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 15, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

        tuned_model = LGBMClassifier(
            **params, verbosity=-1, random_state=self.random_state)
        pipeline = self.pipeline_builder.build(tuned_model)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        return f1_score(y_val, y_pred, average="macro")
