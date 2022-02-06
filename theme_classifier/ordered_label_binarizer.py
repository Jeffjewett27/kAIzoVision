from typing import Iterable
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class OrderedLabelBinarizer():
    """A LabelBinarizer wrapper to guarantee that the classes retain their ordering"""

    def __init__(self, classes: Iterable) -> None:
        self.classes_ = np.array(classes)
        self.classmap_ = {c: i for i, c in enumerate(self.classes_)}
        self.lb_ = LabelBinarizer()
        self.lb_.fit(np.arange(0,len(self.classes_)))

    def get_classes(self):
        return self.classes_
        
    def transform(self, values: np.ndarray) -> np.ndarray:
        """
        Transforms a 1D array of classes to 2D array of onehot equivalents
        """
        indices = np.vectorize(self.classmap_.get)(values)
        return self.lb_.transform(indices)

    def inverse_transform(self, one_hots: np.ndarray) -> np.ndarray:
        """
        Transforms a 2D onehot array to their respective class names
        """
        indices = self.lb_.inverse_transform(one_hots)
        return np.apply_along_axis(lambda i: self.classes_[i], 0, indices)

    def inverse_transform_indices(self, indices: np.ndarray) -> np.ndarray:
        """
        Transforms a 1D array of column indices to their respective class names
        """
        return np.apply_along_axis(lambda i: self.classes_[i], 0, indices)
