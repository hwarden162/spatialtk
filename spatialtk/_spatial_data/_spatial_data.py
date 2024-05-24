from abc import ABC, abstractmethod
import numpy as np
from typing import List

class SpatialData(ABC):
    def __init__(self, coords: np.ndarray, data: np.ndarray, columns: List[str]) -> None:
        assert isinstance(coords, np.ndarray), "coords must be a numpy array"
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        assert isinstance(columns, list), "columns must be a list"
        assert coords.ndim == 2, "coords must be a 2D array"
        assert data.ndim == 2, "data must be a 2D array"
        assert coords.shape[0] == data.shape[0], "coords and data must have the same number of rows"
        assert data.shape[1] == len(columns), "data must have the same number of columns as columns"
        self.coords = coords
        self.data = data
        self.columns = columns
        
    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> np.ndarray:
        assert isinstance(idx, int), "idx must be an integer"
        assert 0 <= idx < len(self), "idx out of bounds"
        return self.coords[idx], self.data[idx]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def get_bounds(self) -> np.ndarray:
        return np.array([self.coords.min(axis=0), self.coords.max(axis=0)])
    
    @abstractmethod
    def filter_spatial_box(self) -> 'SpatialData':
        pass