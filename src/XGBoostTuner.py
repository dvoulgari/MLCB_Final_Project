import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import optuna
from xgboost import XGBClassifier
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from XGBoostBaselineClassifier import (
    XGBoostBaselineClassifier,
    DEFAULT_TEST_SIZE,
    DEFAULT_SEED,
    TRAINING_DATASET
)
from DiskIO import DiskIO
from PipelineBuilder import PipelineBuilder
from MetricsCore import MetricsCalculator
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class XGBoostTunerCV(XGBoostBaselineClassifier):
    def __init__(self, csv_path=TRAINING_DATASET, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_SEED, n_trials=50):
        self.csv_path = csv_path
        self.pipeline_builder = PipelineBuilder()
        self.io = DiskIO("../models")
        self.label_encoder = LabelEncoder()
        super().__init__(csv_path, test_size, random_state)
        self.n_trials = n_trials
        self.best_params = None
        self.n_splits_outer = 5
        self.n_splits_inner = 3

        # Initialize history for plotting
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        # Use the data loading function from the base class
        self.load_and_prepare_data()

    def objective(self, trial, X, y):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
            "random_state": self.random_state,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
        }

        inner_cv = StratifiedKFold(n_splits=self.n_splits_inner, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, val_idx in inner_cv.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]


            model = XGBClassifier(**params)
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_val_fold)
            scores.append(f1_score(y_val_fold, preds, average="weighted"))

        return np.mean(scores)


    def tune_hyperparameters(self):
        outer_cv = StratifiedKFold(
            n_splits=self.n_splits_outer, shuffle=True, random_state=self.random_state)
        
        best_params_list = []
        outer_scores = []

        for train_idx, val_idx in outer_cv.split(self.X_train, self.y_train):
            X_train_fold = self.X_train.iloc[train_idx]
            y_train_fold = self.y_train[train_idx]

            X_val_outer = self.X_train.iloc[val_idx]
            y_val_outer = self.y_train[val_idx]

            # Tune on inner folds
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self.objective(trial, X_train_fold, y_train_fold), 
                        n_trials=self.n_trials)

            best_params = study.best_params.copy()

            # Train with early stopping
            model = xgb.XGBClassifier(
                **best_params,
                random_state=self.random_state,
                eval_metric="mlogloss"  # Moved eval_metric here
            )
            
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_outer, y_val_outer)],
                early_stopping_rounds=20,  # Only early_stopping_rounds goes here
                verbose=False
            )

            # Get best iteration if early stopping was used
            if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                best_params["n_estimators"] = model.best_iteration
            else:
                # Fallback if early stopping didn't trigger
                best_params["n_estimators"] = best_params.get("n_estimators", 100)

            # Evaluate on outer validation fold
            preds = model.predict(X_val_outer)
            score = f1_score(y_val_outer, preds, average="weighted")
            outer_scores.append(score)

            best_params_list.append(best_params)

        # Pick best-performing params from outer loop
        best_idx = np.argmax(outer_scores)
        self.best_params = best_params_list[best_idx]

        print(f"[Nested CV] Outer F1 Scores: {outer_scores}")
        print(f"[Nested CV] Mean Outer F1 Score: {np.mean(outer_scores):.4f}")

    def train_internal_split(self):
        """
        Train the model with the best hyperparameters found during nested CV
        on the internal training set only (approx. 67% of Hao dataset).
        """
        if self.best_params is None:
            raise RuntimeError("Run tune_hyperparameters first.")

        # Initialize empty dict to store results
        self.eval_results = {}
        
        model = xgb.XGBClassifier(
            **self.best_params,
            use_label_encoder=False,
            random_state=self.random_state
        )
        
        # Fit model with evaluation sets
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train),
                    (self.X_test, self.y_test)],
            eval_metric=["mlogloss", "merror"],
            early_stopping_rounds=20,
            verbose=False
        )

        # Store evaluation results AFTER fitting
        self.eval_results = model.evals_result()
        self.pipeline = self.pipeline_builder.build(model)
        self.pipeline.fit(self.X_train, self.y_train)

    def plot_learning_curve(self):
        if not hasattr(self, 'eval_results') or not self.eval_results:
            raise RuntimeError("No evaluation results found. Run train_internal_split first.")

        results = self.eval_results
        
        # Ensure we have the expected keys
        if 'validation_0' not in results or 'validation_1' not in results:
            raise RuntimeError("Evaluation results format not as expected")
        
        # Extract metrics
        train_loss = results['validation_0']['mlogloss']
        val_loss = results['validation_1']['mlogloss']
        train_acc = [1 - x for x in results['validation_0']['merror']]
        val_acc = [1 - x for x in results['validation_1']['merror']]
        
        epochs = range(1, len(train_loss) + 1)
        
        plt.figure(figsize=(12, 8))
        
        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss, label='Train Loss', color='blue')
        plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Log Loss')
        plt.title('XGBoost Log Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_acc, label='Train Accuracy', color='green')
        plt.plot(epochs, val_acc, label='Validation Accuracy', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('XGBoost Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def train_and_save_final_model_on_full_data(self, suffix='final'):
        """
        Train and save the final model on the entire internal dataset (train + test).
        """
        X_full = pd.concat([self.X_train, self.X_test])
        y_full = np.concatenate([self.y_train, self.y_test])

        model = xgb.XGBClassifier(
            **self.best_params,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="mlogloss"
        )

        final_pipeline = self.pipeline_builder.build(model)
        final_pipeline.fit(X_full, y_full)

        # Save both pipeline and label encoder as a tuple
        self.io.save(
            (final_pipeline, self.label_encoder),  # Save as tuple
            name="XGBoost",
            suffix=suffix
        )

    def bootstrap_metrics(self, X, y, n_bootstrap=1000, random_state=None):
        """
        Calculate bootstrap confidence intervals and standard error
        Returns dictionary with metrics, CIs, and standard error
        """
        # Initialize metrics calculator
        metrics_calc = MetricsCalculator(metrics=['Accuracy', 'Recall', 'Specificity', 'PPV', 'NPV', 'F1'])

        # Initialize storage for all metrics
        metric_names = metrics_calc.METRICS
        metrics = {name: [] for name in metric_names}
        
        rng = np.random.RandomState(random_state)
        
        for _ in range(n_bootstrap):
            # Create bootstrap sample
            indices = rng.choice(len(y), size=len(y), replace=True)
            X_bs = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            y_bs = y[indices]

            # Calculate metrics using your MetricsCalculator
            metrics_dict = metrics_calc.compute(y_bs, self.pipeline.predict(X_bs))
            
            # Store all metrics
            for name, value in metrics_dict.items():
                metrics[name].append(value / 100)  # Convert from percentage back to ratio
        
        # Calculate mean, CI, and standard error
        ci = {}
        for metric, values in metrics.items():
            mean = np.mean(values)
            std_error = np.std(values, ddof=1)  # Standard error of the mean (SEM)
            
            ci[metric] = {
                'mean': mean,
                'std_error': std_error,  
                'lower': np.percentile(values, 2.5),
                'upper': np.percentile(values, 97.5)
            }
        
        return ci

    def plot_bootstrap_results(self, ci, title="Bootstrap Results (Mean ± SEM)"):
        """
        Plot metrics with error bars showing ±1 SEM
        """
        metrics = list(ci.keys())
        means = [ci[m]['mean'] for m in metrics]
        errors = [ci[m]['std_error'] for m in metrics]  # Use SEM instead of CI
        
        plt.figure(figsize=(12, 6))
        plt.errorbar(metrics, means, yerr=errors, fmt='o', 
                    capsize=5, capthick=2, markersize=8)
        plt.title(title)
        plt.ylabel('Score (0-1 scale)')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def evaluate_on_external_testset(self, external_csv_path="data/kotliarov.csv", suffix='final', 
                                n_bootstrap=1000, plot_results=True):
        loaded_obj = self.io.load(name="XGBoost", suffix=suffix)

        if isinstance(loaded_obj, tuple) and len(loaded_obj) == 2:
            pipeline, label_encoder = loaded_obj
        else:
            pipeline = loaded_obj
            label_encoder = self.label_encoder

        df_ext = pd.read_csv(external_csv_path)
        X_ext = df_ext[self.X_train.columns.tolist()]
        y_ext = df_ext['label']

        known_labels = set(label_encoder.classes_)
        mask = y_ext.isin(known_labels)
        X_ext = X_ext[mask]
        y_ext = y_ext[mask]
        y_ext_encoded = label_encoder.transform(y_ext)

        y_pred = pipeline.predict(X_ext)

        report = classification_report(
            y_ext_encoded, y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )
        cm = confusion_matrix(y_ext_encoded, y_pred)
        stats = self.metrics_calculator.compute(y_ext_encoded, y_pred)
        
        # Add bootstrap CIs
        bootstrap_ci = self.bootstrap_metrics(X_ext, y_ext_encoded, n_bootstrap=n_bootstrap)
        stats['bootstrap_ci'] = bootstrap_ci
        
        if plot_results:
            self.plot_bootstrap_results(bootstrap_ci, 
                                    title="Bootstrap Confidence Intervals on Test Set")
        
        return report, cm, stats