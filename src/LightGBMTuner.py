import warnings
import optuna
from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, log_loss, accuracy_score
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

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

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

    def plot_training_curves(self):
        if not self.history["train_loss"]:
            print("No training history found. Run tune() first.")
            return

        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Loss Plot
        axs[0].plot(epochs, self.history["train_loss"],
                    marker='o', label="Train Loss")
        axs[0].plot(epochs, self.history["val_loss"],
                    marker='o', label="Validation Loss")
        axs[0].set_title("Log Loss per Trial")
        axs[0].set_xlabel("Trial")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)

        # Accuracy Plot
        axs[1].plot(epochs, self.history["train_acc"],
                    marker='o', label="Train Accuracy")
        axs[1].plot(epochs, self.history["val_acc"],
                    marker='o', label="Validation Accuracy")
        axs[1].set_title("Accuracy per Trial")
        axs[1].set_xlabel("Trial")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

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

        # Compute predictions
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)

        # Compute additional metrics
        train_loss = log_loss(y_train, pipeline.predict_proba(X_train))
        val_loss = log_loss(y_val, pipeline.predict_proba(X_val))
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)

        return f1_score(y_val, y_pred, average="macro")
