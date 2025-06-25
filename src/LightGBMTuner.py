import lightgbm as lgb  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from LightGBMBaselineClassifier import (  # Changed import
    LightGBMBaselineClassifier,
    DEFAULT_TEST_SIZE,
    DEFAULT_SEED,
    TRAINING_DATASET
)
from DiskIO import DiskIO
from PipelineBuilder import PipelineBuilder

warnings.filterwarnings("ignore")

class LightGBMTunerCV(LightGBMBaselineClassifier):  
    def __init__(self, csv_path=TRAINING_DATASET, test_size=DEFAULT_TEST_SIZE, 
                 random_state=DEFAULT_SEED, n_trials=50):
        super().__init__(csv_path, test_size, random_state)
        self.n_trials = n_trials
        self.best_params = None
        self.n_splits_outer = 5
        self.n_splits_inner = 3
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        
        self.load_and_prepare_data()

    def objective(self, trial, X, y):
        params = {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300, step=5),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": -1,
            "max_bin": trial.suggest_int("max_bin", 50, 500),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 1.0)
}

        inner_cv = StratifiedKFold(n_splits=self.n_splits_inner, shuffle=True, 
                                 random_state=self.random_state)
        scores = []

        for train_idx, val_idx in inner_cv.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            preds = model.predict(X_val_fold)
            scores.append(f1_score(y_val_fold, preds, average="weighted"))

        return np.mean(scores)

    def train_final_model(self):
        if self.best_params is None:
            raise RuntimeError("Run tune_hyperparameters first.")
        
        model = lgb.LGBMClassifier(
            **self.best_params,
            random_state=self.random_state
        )
        
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            eval_metric=["multi_logloss", "multi_error"],
            verbose=False
        )

        self.pipeline = self.pipeline_builder.build(model)
        self.pipeline.fit(self.X_train, self.y_train)
        
        self.eval_results = model.evals_result_

    def plot_learning_curve(self):
        if not hasattr(self, "eval_results"):
            raise RuntimeError("No evaluation results found.")
        
        results = self.eval_results
        epochs = len(results['training']['multi_logloss'])
        x_axis = range(epochs)

        plt.figure(figsize=(12, 8))

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(x_axis, results['training']['multi_logloss'], label='Train Loss', color='blue')
        plt.plot(x_axis, results['valid_1']['multi_logloss'], label='Validation Loss', color='orange')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.title('LightGBM Log Loss')
        plt.grid(True)

        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(x_axis, [1-e for e in results['training']['multi_error']], label='Train Accuracy', color='green')
        plt.plot(x_axis, [1-e for e in results['valid_1']['multi_error']], label='Validation Accuracy', color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('LightGBM Accuracy')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def train_and_save_final_model_on_full_data(self, suffix='final'):
        X_full = pd.concat([self.X_train, self.X_test])
        y_full = np.concatenate([self.y_train, self.y_test])

        model = lgb.LGBMClassifier(
            **self.best_params,
            random_state=self.random_state
        )

        final_pipeline = self.pipeline_builder.build(model)
        final_pipeline.fit(X_full, y_full)

        self.save_model(suffix=suffix, pipeline=final_pipeline, label_encoder=self.label_encoder)

    def evaluate_on_external_testset(self, external_csv_path="data/kotliarov.csv", suffix='final'):
        loaded_obj = self.io.load(name="LightGBM", suffix=suffix)  
        
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

        return report, cm, stats