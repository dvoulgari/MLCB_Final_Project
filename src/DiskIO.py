import os
import joblib
from pathlib import Path

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

