import os
import joblib
from pathlib import Path


# class DiskIO:
#     """
#     Handles saving and loading of models or any serializable Python object
#     (e.g., classifiers, pipelines, encoders) using consistent naming conventions.
#     """

#     def __init__(self, base_dir="models"):
#         """
#         Initialize with directory for saving/loading.

#         Args:
#             base_dir (str): Directory path where objects are stored.
#         """
#         if base_dir is None:
#             raise ValueError("base_dir must be provided.")
#         self.base_dir = Path(__file__).resolve().parent.parent / base_dir
#         self.base_dir.mkdir(parents=True, exist_ok=True)

#     def save(self, obj, name, suffix=''):
#         """
#         Save a serializable object to disk.

#         Args:
#             obj: Any joblib-serializable Python object.
#             name (str): Base name (e.g., 'LightGBM', 'label_encoder').
#             suffix (str): Optional descriptor (e.g., 'baseline', 'optimal').
#         """
#         filename = self._generate_filename(name, suffix)
#         print(f"Saving to: {filename.resolve()}")
#         joblib.dump(obj, filename)

#     def load(self, name, suffix=''):
#         """
#         Load a serialized object from disk.

#         Args:
#             name (str): Base name.
#             suffix (str): Optional descriptor.

#         Returns:
#             The loaded Python object.
#         """
#         filename = self._generate_filename(name, suffix)
#         return joblib.load(filename)

#     def _generate_filename(self, name, suffix):
#         """
#         Generate standardized filename.

#         Args:
#             name (str): Base name.
#             suffix (str): Optional suffix.

#         Returns:
#             Path: Full path to the .pkl file.
#         """
#         parts = [name]
#         if suffix:
#             parts.append(suffix)
#         return self.base_dir / f"{'_'.join(parts)}.pkl"

class DiskIO:
    def __init__(self, base_dir="../models"):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DiskIO] Using base directory: {self.base_dir}")
        
    def save(self, obj, name, suffix=''):
        parts = [name]
        if suffix:
            parts.append(suffix)
        filename = self.base_dir / f"{'_'.join(parts)}.pkl"
        joblib.dump(obj, filename)
        print(f"[DiskIO] Saved to: {filename}")

    def load(self, name, suffix=''):
        parts = [name]
        if suffix:
            parts.append(suffix)
        filename = self.base_dir / f"{'_'.join(parts)}.pkl"
        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        print(f"[DiskIO] Loading from: {filename}")
        return joblib.load(filename)

