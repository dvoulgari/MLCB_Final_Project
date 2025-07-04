from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
)
import numpy as np


class MetricsCore:
    def __init__(self, metrics=None):
        self.METRICS = metrics or ['Accuracy',
                                   'Recall', 'Specificity', 'PPV', 'NPV', 'F1']

    def compute(self, y_true, y_pred):
        labels = np.unique(np.concatenate((y_true, y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        results = {}
        for name in self.METRICS:
            if name == 'Accuracy':
                results[name] = round(100 * accuracy_score(y_true, y_pred), 2)
            elif name == 'Recall':
                results[name] = round(
                    100 * recall_score(y_true, y_pred, average='macro'), 2)
            elif name == 'Specificity':
                results[name] = round(100 * np.mean(TN / (TN + FP + 1e-10)), 2)
            elif name == 'PPV':
                results[name] = round(
                    100 * precision_score(y_true, y_pred, average='macro'), 2)
            elif name == 'NPV':
                results[name] = round(100 * np.mean(TN / (TN + FN + 1e-10)), 2)
            elif name == 'F1':
                results[name] = round(
                    100 * f1_score(y_true, y_pred, average='macro'), 2)
        return results


class MetricsCalculator(MetricsCore):
    def compute_from_model(self, model, X, y_true):
        y_pred = model.predict(X)
        return self.compute(y_true, y_pred)
