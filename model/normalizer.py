import numpy as np
from typing import Union
import torch

class PolynomicNormalizer:
    def __init__(self, n_dimensions: int) -> None:
        self.coeffs: list[list[float]] = [[1.0]] * n_dimensions

    @staticmethod
    def spawnBasicFromDataset(dataset: np.ndarray) -> tuple["PolynomicNormalizer", np.ndarray]:
        normalizer = PolynomicNormalizer(len(dataset[0]))

        for i, col in enumerate(dataset.T):
            normalizer.coeffs[i] = [-col.min() /(col.max() - col.min()), 1 / (col.max() - col.min())]

        return normalizer, normalizer.normalizeDataset(dataset)
    
    def normalizeDataset(self,dataset: np.ndarray) -> np.ndarray:
        cols = []
        for i,col in enumerate(dataset.T):
            subtotal = np.zeros_like(col)
            for j,coeff in enumerate(self.coeffs[i]):
                subtotal += coeff * (col ** j)

            cols.append(subtotal)

        return np.array(cols).T
    

    def normalizeColumns(self, column_indices: list[int], dataset: Union[np.ndarray, torch.Tensor, list[np.ndarray]]) -> Union[np.ndarray, torch.Tensor, list[np.ndarray]]:
        cols = []

        dataset = dataset.T if type(dataset) == np.ndarray or type(dataset) == torch.Tensor else dataset
        
        for i, col in enumerate(dataset):
            subtotal = np.zeros_like(col)
            for j,coeff in enumerate(self.coeffs[column_indices[i]]):
                subtotal += coeff * (col ** j)

            cols.append(subtotal)

        #return np.array(cols).T if type(dataset) == np.ndarray or type(dataset) == torch.Tensor else cols
        if type(dataset) == np.ndarray: return np.array(cols).T
        elif type(dataset) == torch.Tensor: return torch.tensor(np.array(cols).T, dtype=torch.float)
        else: return cols
    
    def denormalizeDataset(self, dataset: np.ndarray) -> np.ndarray:
        denormalizedDataset = []
        for i,col in enumerate(dataset.T):
            polynomial = np.poly1d(self.coeffs[i][::-1])
            rootsCol = []
            for j,val in enumerate(col):
                roots = (polynomial - val).roots
                rootsCol += [roots[0]]

            denormalizedDataset.append(np.array(rootsCol))

        return np.array(denormalizedDataset).T
    

    def denormalizeColumns(self, column_indices: list[int], dataset: Union[np.ndarray, torch.Tensor, list[np.ndarray]]) -> Union[np.ndarray, torch.Tensor, list[np.ndarray]]:
        
        cols = []
        dataset = dataset.T if type(dataset) == np.ndarray or type(dataset) == torch.Tensor else dataset

        for i,col in enumerate((dataset.T if type(dataset) == np.ndarray else dataset.detach().numpy())):
            polynomial = np.poly1d(self.coeffs[column_indices[i]][::-1])
            rootsCol = []
            for j,val in enumerate(col):
                roots = (polynomial - val).roots
                rootsCol += [roots[0]]

            cols.append(np.array(rootsCol))
        
        if type(dataset) == np.ndarray: return np.array(cols).T
        elif type(dataset) == torch.Tensor: return torch.tensor(np.array(cols).T, dtype=torch.float)
        else: return cols