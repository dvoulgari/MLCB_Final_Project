from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from MetricsCore import MetricsCore
from PipelineBuilder import PipelineBuilder
from DiskIO import DiskIO


DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.33
MODELS_DIR = "../models"
TRAINING_DATASET = "../data/hao.csv"


class BaseTreeClassifier(ABC):
    """
    Abstract base class for tree-based classifiers.
    Defines common functionality for training, evaluation, persistence,
    and label encoding. Subclasses must implement `build_model()`.
    """

    def __init__(self,
                 csv_path=TRAINING_DATASET,
                 test_size=DEFAULT_TEST_SIZE,
                 random_state=DEFAULT_SEED,
                 models_dir=MODELS_DIR):
        """
        Initialize the classifier with dataset path and configuration.

        Args:
            csv_path (str): Path to the input CSV file with 'label' column.
            test_size (float): Fraction of data to use as test set.
            random_state (int): Seed for reproducibility.
            models_dir (str): Path to directory for saving models and encoders.
        """
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.metrics_calculator = MetricsCore()
        self.pipeline_builder = PipelineBuilder()
        self.io = DiskIO(models_dir)

        self.load_and_prepare_data()

    def load_and_prepare_data(self):
        """
        Loads the CSV, splits into train/test sets, and applies label encoding.
        """
        data = pd.read_csv(self.csv_path, index_col=0)
        self.features = data.drop(columns=["label"])
        labels_raw = data["label"]

        # Stratified train-test split to preserve label proportions
        self.X_train, self.X_test, self.y_train_raw, self.y_test_raw = train_test_split(
            self.features, labels_raw,
            test_size=self.test_size,
            stratify=labels_raw,
            random_state=self.random_state
        )

        # Fit label encoder only on training labels to avoid data leakage
        self.label_encoder.fit(self.y_train_raw)
        self.y_train = self.label_encoder.transform(self.y_train_raw)
        self.y_test = self.label_encoder.transform(self.y_test_raw)

    @abstractmethod
    def build_model(self):
        """
        Subclasses must override this method to return an untrained model instance.
        """
        pass

    def train(self):
        """
        Builds and trains the pipeline on training data.
        """
        model = self.build_model()
        self.pipeline = self.pipeline_builder.build(model)
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evaluates the trained model on the test set.

        Returns:
            report (dict): Classification report (per-class and aggregate).
            cm (ndarray): Confusion matrix.
            stats (dict): Custom metrics computed using MetricsCore.
        """
        y_pred = self.pipeline.predict(self.X_test)
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        cm = confusion_matrix(self.y_test, y_pred)
        stats = self.metrics_calculator.compute(self.y_test, y_pred)
        return report, cm, stats

    def plot_confusion_matrix(self, cm):
        """
        Plots the confusion matrix as a heatmap.

        Args:
            cm (ndarray): Confusion matrix from evaluation.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            cmap='Blues'
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()

    def save_model(self, suffix='', pipeline=None):
        """
        Saves the trained pipeline to disk using naming convention.

        Args:
            suffix (str): Optional suffix to distinguish model versions.
            pipeline (Pipeline): Optional pipeline to save (defaults to self.pipeline).
        """
        if pipeline is None:
            pipeline = self.pipeline

        if pipeline is None:
            raise ValueError("Pipeline must be available to save.")

        model_name = self.__class__.__name__.replace("Classifier", "")
        self.io.save(pipeline, name=model_name, suffix=suffix)

    def load_model(self, suffix=''):
        """
        Loads a previously saved pipeline from disk.

        Args:
            suffix (str): Optional suffix for model versioning.

        Returns:
            pipeline (Pipeline): Loaded pipeline object.
        """
        model_name = self.__class__.__name__.replace("Classifier", "")
        pipeline = self.io.load(name=model_name, suffix=suffix)
        return pipeline

    def save_label_encoder(self, suffix=None):
        """
        Saves the fitted label encoder, tied to the dataset name.

        Args:
            suffix (str): Optional suffix to distinguish encoders.
        """
        name = f"label_encoder_{Path(self.csv_path).stem}"
        if suffix:
            name += f"_{suffix}"
        self.io.save(self.label_encoder, name=name)

    def load_label_encoder(self, suffix=None):
        """
        Loads a previously saved label encoder.

        Args:
            suffix (str): Optional suffix to distinguish encoders.

        Returns:
            LabelEncoder: Loaded label encoder object.
        """
        name = f"label_encoder_{Path(self.csv_path).stem}"
        if suffix:
            name += f"_{suffix}"
        return self.io.load(name=name)
