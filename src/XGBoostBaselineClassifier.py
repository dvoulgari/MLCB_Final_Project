from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PipelineBuilder import PipelineBuilder
from MetricsCore import MetricsCore
from DiskIO import DiskIO

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.33
MODELS_DIR = "../models"
TRAINING_DATASET = "../data/hao.csv"
TESTING_DATASET = "../data/kotliarov.csv"

class XGBoostBaselineClassifier:
    METRICS = ['Accuracy', 'Recall', 'Specificity', 'PPV', 'NPV', 'F1']

    def __init__(self, csv_path=TRAINING_DATASET, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_SEED):
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.base_dir = Path(MODELS_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_calculator = MetricsCore()
        self.pipeline_builder = PipelineBuilder()
        self.io = DiskIO(MODELS_DIR)

    def load_and_prepare_data(self):
            data = pd.read_csv(self.csv_path, index_col=0)

            self.features = data.drop(columns=["label"])
            labels_raw = data["label"]

            # Split data *before* fitting the LabelEncoder - Split hao.csv internally here for tuning/validation
            self.X_train, self.X_test, self.y_train_raw, self.y_test_raw = train_test_split(  # X_train, y_train will be used for hyperparameter tuning and model fitting,
                                                                                            #  X_test, y_test for validation within hao.csv to evaluate tuning (NOT final test)
                self.features, labels_raw,
                test_size=self.test_size,
                stratify=labels_raw, # Stratify on raw labels
                random_state=self.random_state
            )

            # Fit LabelEncoder only on training labels
            self.label_encoder.fit(self.y_train_raw) 

            # Transform both training and testing labels
            self.y_train = self.label_encoder.transform(self.y_train_raw)
            self.y_test = self.label_encoder.transform(self.y_test_raw)

    def train(self):
        model = XGBClassifier(random_state=self.random_state)

        self.pipeline = self.pipeline_builder.build(model)
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.pipeline.predict(self.X_test)
        report = classification_report(
            self.y_test, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)
        stats = self.metrics_calculator.compute(self.y_test, y_pred)
        return report, cm, stats

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()

    def save_model(self, suffix='baseline', pipeline=None, label_encoder=None):
        if pipeline is None or label_encoder is None:
            raise ValueError("pipeline and label_encoder must be provided")

        self.io.save(pipeline, name="XGBoost", suffix=suffix)
        self.io.save(label_encoder, name="label_encoder", suffix=suffix)

    def load_model(self, suffix='baseline', base_dir='models'):
        from DiskIO import DiskIO
        io = DiskIO(base_dir=base_dir)

        pipeline = io.load(name="XGBoost", suffix=suffix)
        label_encoder = io.load(name="label_encoder_XGBoost", suffix=suffix)
        return pipeline, label_encoder