from sklearn.ensemble import RandomForestClassifier
from BaseTreeClassifier import BaseTreeClassifier


class RFClassifier(BaseTreeClassifier):
    """
    Random Forest implementation of the BaseTreeClassifier.

    Inherits:
        BaseTreeClassifier: Provides training, evaluation, and serialization logic.

    This class only needs to implement the `build_model()` method
    to specify the estimator to be used in the pipeline.
    """

    def build_model(self):
        """
        Instantiate and return a RandomForestClassifier with predefined settings.

        Returns:
            RandomForestClassifier: A scikit-learn classifier with fixed random seed and full parallelization.
        """
        return RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1
        )
