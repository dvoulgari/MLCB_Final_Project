import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import optuna
from xgboost import XGBClassifier
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
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class XGBoostTunerCV(XGBoostBaselineClassifier):
    def __init__(self, csv_path=TRAINING_DATASET, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_SEED, n_trials=50):
        self.csv_path = csv_path
        self.pipeline_builder = PipelineBuilder()
        self.io = DiskIO()
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
        outer_cv = StratifiedKFold(n_splits=self.n_splits_outer, shuffle=True, random_state=self.random_state)
        best_params_list = []

        for train_idx, val_idx in outer_cv.split(self.X_train, self.y_train): # use only X_train, y_train for tuning (X_test, y_test untouched)
            X_train_fold = self.X_train.iloc[train_idx]
            y_train_fold = self.y_train[train_idx]

            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self.objective(trial, X_train_fold, y_train_fold), n_trials=self.n_trials)

            best_params_list.append(study.best_params)

        self.best_params = max(best_params_list, key=lambda p: best_params_list.count(p))

    def train_final_model(self):
        if self.best_params is None:
            raise RuntimeError("Run tune_hyperparameters first.")
        
        model = XGBClassifier(
            **self.best_params, 
            random_state=self.random_state, 
            use_label_encoder=False, 
            eval_metric=["mlogloss", "merror"]
            )
        
        model.fit(
            self.X_train, self.y_train, # fit on the entire training set (67% of hao.csv)
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)], # use X_test, y_test for monitoring validation during training - logging learning curves
            verbose=False
            )

        self.pipeline = self.pipeline_builder.build(model)
        self.pipeline.fit(self.X_train, self.y_train)  # Fit the entire pipeline

        self.eval_results = model.evals_result()

    def evaluate_final_model(self):
        y_pred = self.pipeline.predict(self.X_test) # final performance evaluation on the untouched X_test, y_test (33% of hao.csv)

        report = classification_report(
            self.y_test, y_pred, target_names=self.label_encoder.classes_, output_dict=True
            )
        
        cm = confusion_matrix(self.y_test, y_pred)

        stats = self.metrics_calculator.compute(self.y_test, y_pred)

        return report, cm, stats

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,  
            yticklabels=self.label_encoder.classes_   
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self):
        if not hasattr(self, "eval_results"):
            raise RuntimeError("No evaluation results found. Train the model with eval_set to get learning curves.")
        
        results = self.eval_results
        epochs = len(results['validation_0']['mlogloss'])
        x_axis = range(epochs)

        train_loss = results['validation_0']['mlogloss']
        val_loss = results['validation_1']['mlogloss']
        train_acc = [1 - e for e in results['validation_0']['merror']]
        val_acc = [1 - e for e in results['validation_1']['merror']]

        plt.figure(figsize=(12, 8))

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(x_axis, train_loss, label='Train Loss', color='blue', linestyle='-')
        plt.plot(x_axis, val_loss, label='Validation Loss', color='orange', linestyle='--')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.title('XGBoost Log Loss')
        plt.grid(True)

        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(x_axis, train_acc, label='Train Accuracy', color='green', linestyle='-')
        plt.plot(x_axis, val_acc, label='Validation Accuracy', color='red', linestyle='--')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('XGBoost Accuracy')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    
    # def train_and_save_final_model_on_full_data(self, suffix='final'):
    #     # Combine train and test data for final training
    #     X_full = pd.concat([self.X_train, self.X_test])
    #     y_full = pd.concat([pd.Series(self.y_train), pd.Series(self.y_test)])

    #     # Create fresh XGBClassifier with best params (unfitted)
    #     model = XGBClassifier(
    #         **self.best_params,
    #         random_state=self.random_state,
    #         use_label_encoder=False,
    #         eval_metric="mlogloss"
    #     )

    #     # Build the pipeline with the fresh model
    #     final_pipeline = self.pipeline_builder.build(model)

    #     # Fit the entire pipeline (scaler + model) on full data
    #     final_pipeline.fit(X_full, y_full)

    #     # Save the fitted pipeline and label encoder
    #     self.save_model(suffix=suffix, pipeline=final_pipeline, label_encoder=self.label_encoder)


    def train_and_save_final_model_on_full_data(self, suffix='final'):
        # Combine train and test data for final training
        X_full = pd.concat([self.X_train, self.X_test])
        y_full = np.concatenate([self.y_train, self.y_test])

        # Create fresh XGBClassifier with best params
        model = XGBClassifier(
            **self.best_params,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="mlogloss"
        )

        # Build and fit the pipeline
        final_pipeline = self.pipeline_builder.build(model)
        final_pipeline.fit(X_full, y_full)

        # Save the fitted pipeline and label encoder
        self.save_model(suffix=suffix, pipeline=final_pipeline, label_encoder=self.label_encoder)

    # def evaluate_on_external_testset(self, external_csv_path="data/kotliarov.csv", suffix='final'):
    #     df = pd.read_csv(external_csv_path)
    #     X_ext = df.drop(columns=["label"])
    #     y_ext = df["label"]

    #     pipeline, label_encoder = self.load_model(suffix=suffix)

    #     # Show any unseen labels
    #     known_labels = set(label_encoder.classes_)
    #     test_labels = set(y_ext)
    #     unseen_labels = test_labels - known_labels
    #     if unseen_labels:
    #         print(f"[WARNING] Unseen labels in external test set: {unseen_labels}")

    #     # Filter out unseen labels 
    #     mask = y_ext.isin(known_labels)
    #     X_ext = X_ext[mask]
    #     y_ext = y_ext[mask]

    #     y_ext_encoded = label_encoder.transform(y_ext)
    #     y_pred = pipeline.predict(X_ext)

    #     # Generate classification report, confusion matrix, and other stats
    #     report = classification_report(
    #         y_ext_encoded, y_pred,
    #         target_names=label_encoder.classes_,
    #         output_dict=True
    #     )
    #     cm = confusion_matrix(y_ext_encoded, y_pred)
    #     stats = self.metrics_calculator.compute(y_ext_encoded, y_pred)

    #     return report, cm, stats

    def evaluate_on_external_testset(self, external_csv_path="data/kotliarov.csv", suffix='final'):
        # Load pipeline and label encoder
        loaded_obj = self.io.load(name="XGBoost", suffix=suffix)
        
        # Check if we got a tuple (pipeline, label_encoder) or just pipeline
        if isinstance(loaded_obj, tuple) and len(loaded_obj) == 2:
            pipeline, label_encoder = loaded_obj
        else:
            # Backward compatibility - if only pipeline was saved
            pipeline = loaded_obj
            label_encoder = self.label_encoder  # use the instance's label encoder

            # Load external test data - ensure same columns as training
            df_ext = pd.read_csv(external_csv_path)

            # Get the feature names used during training
            training_features = self.X_train.columns.tolist()
            
            # Make sure test data has exactly these columns
            X_ext = df_ext[training_features]
            y_ext = df_ext['label']
            
            # Handle unseen labels
            known_labels = set(label_encoder.classes_)
            test_labels = set(y_ext)
            if unseen_labels := test_labels - known_labels:
                print(f"[WARNING] Unseen labels: {unseen_labels}")
                mask = y_ext.isin(known_labels)
                X_ext = X_ext[mask]
                y_ext = y_ext[mask]
            
            y_ext_encoded = label_encoder.transform(y_ext)

        # Predict and evaluate
        try:
            y_pred = pipeline.predict(X_ext)
        except NotFittedError:
            raise RuntimeError("Loaded pipeline is not fitted. Please train and save the model correctly.")

        # Generate metrics
        report = classification_report(
            y_ext_encoded, y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )
        cm = confusion_matrix(y_ext_encoded, y_pred)
        stats = self.metrics_calculator.compute(y_ext_encoded, y_pred)

        return report, cm, stats
