from abc import ABCMeta, abstractmethod

import numpy as np
import torch


class DataToTensor(ABCMeta):
    @abstractmethod
    def __call__(cls, data: any) -> torch.Tensor:
        raise NotImplementedError('The method is not implemented')


class NPArrayToTensor(DataToTensor):
    def __call__(cls, data: np.array) -> torch.Tensor:
        tensor = torch.from_numpy(data)
        tensor = tensor[np.newaxis, :]
        tensor = tensor.permute(0, 3, 2, 1)
        return tensor
