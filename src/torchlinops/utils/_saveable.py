from dataclasses import dataclass, field, asdict
from pathlib import Path
try:
    import cPickle as pickle
except ImportError:
    import pickle

__all__ = ['Saveable']

@dataclass
class Saveable:
    def save(self, filepath: Path):
        with open(filepath, 'wb') as f:
            pickle.dump(asdict(self), f)

    @classmethod
    def load(cls, filepath: Path):
        with open(cachefile, 'rb') as f:
            data_dict = pickle.load(f)
        return cls(**data_dict)
