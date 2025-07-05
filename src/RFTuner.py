from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from BaseTreeTunerCV import BaseTreeTunerCV


class RFTuner(BaseTreeTunerCV):
    """
    Random Forest tuner using Optuna and learning curve visualization.

    Inherits:
        BaseTreeTunerCV: Provides cross-validation and Optuna-based hyperparameter optimization.

    Responsibilities:
    - Define search space for Random Forest via Optuna
    - Instantiate a model with given hyperparameters
    - Visualize learning curves based on F1-weighted score
    """

    def suggest_params(self, trial):
        """
        Define the hyperparameter search space for Optuna trials.

        Args:
            trial (optuna.Trial): Current trial object used to suggest parameters.

        Returns:
            dict: Dictionary of suggested hyperparameters for RandomForestClassifier.
        """
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": self.random_state,
            "n_jobs": -1
        }

    def build_model(self, **params):
        """
        Instantiate a RandomForestClassifier using provided hyperparameters.

        Args:
            **params: Arbitrary keyword arguments for the classifier.

        Returns:
            RandomForestClassifier: Configured instance of RandomForestClassifier.
        """
        return RandomForestClassifier(**params)

    def plot_learning_curve(self, scoring="f1_weighted", cv=5):
        """
        Plot training and validation learning curves for the best Random Forest model.

        Args:
            scoring (str): Scoring metric used in learning curve (default: "f1_weighted").
            cv (int): Number of cross-validation folds (default: 5).

        Raises:
            RuntimeError: If `best_params` is not set (i.e., tuning hasn't run yet).
        """
        if self.best_params is None:
            raise RuntimeError(
                "Run tune_hyperparameters or set best_params first.")

        model = self.build_model(**self.best_params)

        train_sizes, train_scores, val_scores = learning_curve(
            estimator=model,
            X=self.X_train,
            y=self.y_train,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        # Mean and std for error bands
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        # Plot 1: Score
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(train_sizes, train_scores_mean,
                 label='Train Score', color='blue')
        plt.fill_between(train_sizes,
                         train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std,
                         alpha=0.2, color='blue')
        plt.plot(train_sizes, val_scores_mean,
                 label='Validation Score', color='orange')
        plt.fill_between(train_sizes,
                         val_scores_mean - val_scores_std,
                         val_scores_mean + val_scores_std,
                         alpha=0.2, color='orange')
        plt.xlabel("Training Set Size")
        plt.ylabel("F1 Weighted Score")
        plt.title("Random Forest Learning Curve (F1 Score)")
        plt.legend()
        plt.grid(True)

        # Plot 2: Std deviation
        plt.subplot(2, 1, 2)
        plt.plot(train_sizes, train_scores_std,
                 label='Train Score Std', color='green')
        plt.plot(train_sizes, val_scores_std,
                 label='Validation Score Std', color='red')
        plt.xlabel("Training Set Size")
        plt.ylabel("Score Std Deviation")
        plt.title("Score Stability")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
