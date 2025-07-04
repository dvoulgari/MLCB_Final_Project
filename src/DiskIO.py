import joblib
from pathlib import Path


class DiskIO:
    """
    Handles saving and loading of models or any serializable Python object
    (e.g., classifiers, pipelines, encoders) using consistent naming conventions.
    """

    def __init__(self, models_dir):
        """
        Initialize with directory for saving/loading.

        Args:
            models_dir (str): Directory path where objects are stored.
        """
        if models_dir is None:
            raise ValueError("models_dir must be provided.")
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save(self, obj, name, suffix=''):
        """
        Save a serializable object to disk.

        Args:
            obj: Any joblib-serializable Python object.
            name (str): Base name (e.g., 'LightGBM', 'label_encoder').
            suffix (str): Optional descriptor (e.g., 'baseline', 'optimal').
        """
        filename = self._generate_filename(name, suffix)
        joblib.dump(obj, filename)

    def load(self, name, suffix=''):
        """
        Load a serialized object from disk.

        Args:
            name (str): Base name.
            suffix (str): Optional descriptor.

        Returns:
            The loaded Python object.
        """
        filename = self._generate_filename(name, suffix)
        return joblib.load(filename)

    def _generate_filename(self, name, suffix):
        """
        Generate standardized filename.

        Args:
            name (str): Base name.
            suffix (str): Optional suffix.

        Returns:
            Path: Full path to the .pkl file.
        """
        parts = [name]
        if suffix:
            parts.append(suffix)
        return self.models_dir / f"{'_'.join(parts)}.pkl"
