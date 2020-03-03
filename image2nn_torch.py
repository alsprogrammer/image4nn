from abc import ABCMeta, abstractmethod

import numpy as np
import torch


class DataToTensor(ABCMeta):
    @abstractmethod
    def __call__(cls, data: any) -> torch.Tensor:
        pass


class NPArrayToTensor(DataToTensor):
    def __call__(cls, data: np.array) -> torch.Tensor:
        if not data:
            raise ValueError('No data array to convert to tensor')
            
        if not isinstance(data, np.array):
            raise ValueError('The data you are passing is not a numpy array')
            
        if len(data.shape) != 4:
            raise ValueError('The size of the array you are trying to convert is incorrect: {}, instead of 4'.format(data.shape))
        
        tensor = torch.from_numpy(data)
        tensor = tensor[np.newaxis, :]
        tensor = tensor.permute(0, 3, 2, 1)
        return tensor
