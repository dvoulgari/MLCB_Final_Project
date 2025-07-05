from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
)
import numpy as np
from scipy.stats import bootstrap


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

    def compute_with_ci(self, y_true, y_pred, n_resamples=1000, alpha=0.05):
        """
        Compute metrics with confidence intervals using bootstrapping.
        Returns a dict of dicts: {metric: {value, ci_low, ci_high, sem}}
        """
        results = {}
        rng = np.random.default_rng(42)  # for reproducibility

        for name in self.METRICS:
            if name == 'Accuracy':
                def fn(yt, yp): return accuracy_score(yt, yp)
            elif name == 'Recall':
                def fn(yt, yp): return recall_score(yt, yp, average='macro')
            elif name == 'Specificity':
                def fn(yt, yp): return np.mean(
                    np.diag(confusion_matrix(yt, yp)) /
                    (np.sum(confusion_matrix(yt, yp), axis=1) + 1e-10)
                )
            elif name == 'PPV':
                def fn(yt, yp): return precision_score(yt, yp, average='macro')
            elif name == 'NPV':
                def fn(yt, yp): return np.mean(
                    np.diag(confusion_matrix(yt, yp)) /
                    (np.sum(confusion_matrix(yt, yp), axis=0) + 1e-10)
                )
            elif name == 'F1':
                def fn(yt, yp): return f1_score(yt, yp, average='macro')
            else:
                continue

            value = fn(y_true, y_pred)
            res = bootstrap((y_true, y_pred), fn,
                            n_resamples=n_resamples,
                            confidence_level=1-alpha,
                            random_state=rng,
                            method="basic")
            ci_low = res.confidence_interval.low
            ci_high = res.confidence_interval.high
            sem = np.std([fn(y_true, y_pred)], ddof=1) / np.sqrt(len(y_true))

            results[name] = {
                "value": round(100 * value, 2),
                "ci_low": round(100 * ci_low, 2),
                "ci_high": round(100 * ci_high, 2),
                "sem": round(100 * sem, 2)
            }

        return results


class MetricsCalculator(MetricsCore):
    def compute_from_model(self, model, X, y_true, with_ci=False, **kwargs):
        y_pred = model.predict(X)

        if with_ci:
            # Use inherited method that returns dict of {metric: {value, ci_low, ci_high, sem}}
            ci_results = self.compute_with_ci(y_true, y_pred, **kwargs)

            metrics = {k: v["value"] for k, v in ci_results.items()}
            error_bars = {k: (v["value"] - v["ci_low"], v["ci_high"] - v["value"])
                          for k, v in ci_results.items()}
            return metrics, error_bars

        # Default simple scores
        return self.compute(y_true, y_pred)
