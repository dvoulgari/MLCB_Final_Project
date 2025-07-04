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
    Child classes must implement the `build_model` method.
    """

    def __init__(self,
                 csv_path=TRAINING_DATASET,
                 test_size=DEFAULT_TEST_SIZE,
                 random_state=DEFAULT_SEED,
                 models_dir=MODELS_DIR):

        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.metrics_calculator = MetricsCore()
        self.pipeline_builder = PipelineBuilder()
        self.io = DiskIO(models_dir)

        self.load_and_prepare_data()

    def load_and_prepare_data(self):
        data = pd.read_csv(self.csv_path, index_col=0)
        self.features = data.drop(columns=["label"])
        labels_raw = data["label"]

        # Split data
        self.X_train, self.X_test, self.y_train_raw, self.y_test_raw = train_test_split(
            self.features, labels_raw,
            test_size=self.test_size,
            stratify=labels_raw,
            random_state=self.random_state
        )

        # Fit and transform labels
        self.label_encoder.fit(self.y_train_raw)
        self.y_train = self.label_encoder.transform(self.y_train_raw)
        self.y_test = self.label_encoder.transform(self.y_test_raw)

    @abstractmethod
    def build_model(self):
        """
        Subclasses must return an instance of a classifier with relevant parameters set.
        """
        pass

    def train(self):
        model = self.build_model()
        self.pipeline = self.pipeline_builder.build(model)
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
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
        if pipeline is None:
            pipeline = self.pipeline

        if pipeline is None:
            raise ValueError("Pipeline must be available to save.")

        model_name = self.__class__.__name__.replace("Classifier", "")
        self.io.save(pipeline, name=model_name, suffix=suffix)

    def load_model(self, suffix=''):
        model_name = self.__class__.__name__.replace("Classifier", "")
        pipeline = self.io.load(name=model_name, suffix=suffix)
        return pipeline

    def save_label_encoder(self, suffix=None):
        """
        Save the label encoder based on the dataset filename and optional suffix.
        """
        name = f"label_encoder_{Path(self.csv_path).stem}"
        if suffix:
            name += f"_{suffix}"
        self.io.save(self.label_encoder, name=name)

    def load_label_encoder(self, suffix=None):
        """
        Load the label encoder saved for this dataset.
        """
        name = f"label_encoder_{Path(self.csv_path).stem}"
        if suffix:
            name += f"_{suffix}"
        return self.io.load(name=name)
