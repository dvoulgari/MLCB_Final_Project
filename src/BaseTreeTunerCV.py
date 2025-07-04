from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from BaseTreeClassifier import BaseTreeClassifier


class BaseTreeTunerCV(BaseTreeClassifier, ABC):
    def __init__(self, csv_path, test_size=0.33, random_state=42,
                 models_dir="../models", n_trials=20,
                 n_splits_outer=5, n_splits_inner=3):
        super().__init__(csv_path, test_size, random_state, models_dir)
        self.n_trials = n_trials
        self.n_splits_outer = n_splits_outer
        self.n_splits_inner = n_splits_inner
        self.best_params = None
        self.eval_results = {}

    @abstractmethod
    def suggest_params(self, trial):
        pass

    @abstractmethod
    def build_model(self, **params):
        pass

    def objective(self, trial, X, y):
        params = self.suggest_params(trial)
        skf = StratifiedKFold(n_splits=self.n_splits_inner,
                              shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y[val_idx]

            model = self.build_model(**params)
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_val_fold)
            score = f1_score(y_val_fold, preds, average="weighted")
            scores.append(score)

        return np.mean(scores)

    def tune_hyperparameters(self):
        import optuna
        best_params_list = []
        outer_scores = []

        outer_cv = StratifiedKFold(
            n_splits=self.n_splits_outer, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in outer_cv.split(self.X_train, self.y_train):
            X_train_fold = self.X_train.iloc[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_outer = self.X_train.iloc[val_idx]
            y_val_outer = self.y_train[val_idx]

            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self.objective(trial, X_train_fold, y_train_fold),
                           n_trials=self.n_trials)

            best_params = study.best_params
            model = self.build_model(**best_params)
            model.fit(X_train_fold, y_train_fold)

            preds = model.predict(X_val_outer)
            score = f1_score(y_val_outer, preds, average="weighted")
            outer_scores.append(score)
            best_params_list.append(best_params)

        best_idx = int(np.argmax(outer_scores))
        self.best_params = best_params_list[best_idx]

        print(f"[Nested CV] Outer F1 Scores: {outer_scores}")
        print(f"[Nested CV] Mean Outer F1 Score: {np.mean(outer_scores):.4f}")

    def train_internal_split(self):
        if self.best_params is None:
            raise RuntimeError("Run tune_hyperparameters first.")
        model = self.build_model(**self.best_params)
        model.fit(self.X_train, self.y_train)
        self.pipeline = self.pipeline_builder.build(model)
        self.pipeline.fit(self.X_train, self.y_train)

    def train_and_save_final_model_on_full_data(self, suffix='final'):
        X_full = pd.concat([self.X_train, self.X_test])
        y_full = np.concatenate([self.y_train, self.y_test])

        model = self.build_model(**self.best_params)
        final_pipeline = self.pipeline_builder.build(model)
        final_pipeline.fit(X_full, y_full)

        self.save_model(suffix=suffix, pipeline=final_pipeline,
                        label_encoder=self.label_encoder)

    def evaluate_on_external_testset(self, external_csv_path, suffix='final'):
        pipeline, label_encoder = self.load_model(suffix=suffix)

        df_ext = pd.read_csv(external_csv_path)
        X_ext = df_ext[self.X_train.columns.tolist()]
        y_ext = df_ext["label"]

        mask = y_ext.isin(label_encoder.classes_)
        X_ext = X_ext[mask]
        y_ext = y_ext[mask]
        y_ext_encoded = label_encoder.transform(y_ext)
        y_pred = pipeline.predict(X_ext)

        report = classification_report(
            y_ext_encoded, y_pred, target_names=label_encoder.classes_, output_dict=True)
        cm = confusion_matrix(y_ext_encoded, y_pred)
        stats = self.metrics_calculator.compute(y_ext_encoded, y_pred)
        return report, cm, stats
