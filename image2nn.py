from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
import torch
import cv2


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


class ImageArrayGetter(ABCMeta):
    @abstractmethod
    def __call__(cls) -> np.array:
        pass


class CV2CamImageArrayGetter(ImageArrayGetter):
    def __init__(cls, camera):
        super().__init__(camera)
        cls._camera = camera

    def __call__(cls) -> np.array:
        ret, image = cls._camera.read()
        if not ret:
            raise RuntimeError("Cannot get mage from the camera")

        return image


class FileImageArrayGetter(ImageArrayGetter):
    def __init__(cls, resizer: callable):
        super().__init__(resizer)
        cls._resizer = resizer

    def __call__(cls, filename: str, image_size: Tuple[int, int]) -> np.array:
        image = cv2.imread(filename)
        image = cls._resizer(image, image_size)

        return image
