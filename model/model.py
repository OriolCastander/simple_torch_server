import numpy as np
import torch

from typing import Union
from model.normalizer import PolynomicNormalizer

class Model(torch.nn.Module):

    def __init__(self, input_dim: int, hiddenDims: list[int], output_dim: int) -> None:

        super(Model,self).__init__()
        layers = [torch.nn.Linear(input_dim, hiddenDims[0]), torch.nn.ReLU()]
        for i in range(0, len(hiddenDims) - 1):
            layers += [torch.nn.Linear(hiddenDims[i], hiddenDims[i+1]), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hiddenDims[-1], output_dim)]

        self.network = torch.nn.Sequential(*layers)
        
        self.loss_function = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)


    @staticmethod
    def load(path: str, input_dim: int, hiddenDims: list[int], output_dim: int) -> "Model":
        model = Model(input_dim, hiddenDims, output_dim)
        model.load_state_dict(torch.load(path))
        return model
    
    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    def train(self, n_epochs: int, train_dataset: torch.Tensor, eval_dataset: torch.Tensor) -> tuple[list[float], list[float]]:
        BATCH_SIZE = 128
        
        trainLosses, evalLosses = [], []

        for _ in range(n_epochs):
            train_data = train_dataset[torch.randperm(len(train_dataset))[:BATCH_SIZE]]

            self.network.train()
            predictions = self.network(train_data.T[:-1].T)
            loss = self.loss_function(predictions, train_data.T[-1:].T)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            trainLosses += [loss.item()]


            self.network.eval()
            totalEvalLoss: list[float] = []
                
            for j in range(int(len(eval_dataset) / BATCH_SIZE)):
                eval_data = eval_dataset[j * BATCH_SIZE : (j+1) * BATCH_SIZE]
                    
                predictions = self.network(eval_data.T[:-1].T)
                loss = self.loss_function(predictions, eval_data.T[-1:].T)
                totalEvalLoss += [loss.item()]

            evalLosses += [np.mean(totalEvalLoss)]

        return trainLosses, evalLosses
    
    def predict(self, input: torch.Tensor, normalizer: Union["PolynomicNormalizer", None] = None) -> torch.Tensor:
        column_indices = list(range(len(input[0])))
        self.network.eval()
        output = self.network(input)

        output = output if normalizer is None else normalizer.denormalizeColumns(column_indices, output)
        return output