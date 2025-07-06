from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
)
import numpy as np
from scipy.stats import bootstrap


class MetricsCore:
    """
    Base class for computing classification metrics and confidence intervals (CIs)
    from predicted and true labels. Supports Accuracy, Recall, Specificity, PPV, NPV, F1.
    """

    def __init__(self, metrics=None):
        """
        Initialize the metrics calculator.

        Args:
            metrics (list of str): Optional list of metric names to compute.
                                   Defaults to all supported metrics.
        """
        self.METRICS = metrics or ['Accuracy',
                                   'Recall', 'Specificity', 'PPV', 'NPV', 'F1']

    def compute(self, y_true, y_pred):
        """
        Compute classification metrics as percentages.

        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.

        Returns:
            dict: Metric name → score (0–100 range).
        """

        # Ensure all classes are included
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
        Compute metrics with 95% confidence intervals via bootstrapping.

        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.
            n_resamples (int): Number of bootstrap resamples.
            alpha (float): Significance level (default 0.05 for 95% CI).

        Returns:
            dict: {
                metric_name: {
                    'value': score in percent,
                    'ci_low': lower bound of CI,
                    'ci_high': upper bound of CI,
                    'sem': standard error of the mean
                },
                ...
            }
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
                            confidence_level=1 - alpha,
                            random_state=rng,
                            method="basic",
                            vectorized=False,
                            paired=True,
                            axis=-1)
            ci_low = res.confidence_interval.low
            ci_high = res.confidence_interval.high
            sem = res.standard_error

            results[name] = {
                "value": round(100 * value, 2),
                "ci_low": round(100 * ci_low, 2),
                "ci_high": round(100 * ci_high, 2),
                "sem": round(100 * sem, 2)
            }

        return results


class MetricsCalculator(MetricsCore):
    """
    Extension of MetricsCore that supports applying metrics directly to scikit-learn models.
    """

    def compute_from_model(self, model, X, y_true, with_ci=False, **kwargs):
        """
        Compute metrics from a fitted classifier and input data.

        Args:
            model: Fitted scikit-learn classifier with `.predict()` method.
            X (array-like): Input features.
            y_true (array-like): True labels.
            with_ci (bool): If True, return metrics with 95% confidence intervals.
            **kwargs: Additional arguments to pass to `compute_with_ci`.

        Returns:
            If with_ci=False:
                dict of metric_name → score
            If with_ci=True:
                tuple: (metrics_dict, error_bars_dict)
                where:
                    - metrics_dict = metric_name → score
                    - error_bars_dict = metric_name → (lower_error, upper_error)
        """
        y_pred = model.predict(X)

        if with_ci:
            # Use inherited method that returns full CI dict
            ci_results = self.compute_with_ci(y_true, y_pred, **kwargs)

            metrics = {k: v["value"] for k, v in ci_results.items()}
            error_bars = {k: (v["value"] - v["ci_low"], v["ci_high"] - v["value"])
                          for k, v in ci_results.items()}
            sems = {k: v["sem"] for k, v in ci_results.items()}

            return metrics, error_bars, sems

        # Default: only compute point estimates
        # NOTE: This method is kept for backwards compatibility.
        # Prefer `compute_with_ci()` in new workflows.
        return self.compute(y_true, y_pred)
