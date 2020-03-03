from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
import torch
import cv2


class ImageArrayGetter(ABCMeta):
    @abstractmethod
    def __call__(cls) -> np.array:
        pass


class CV2CamImageArrayGetter(ImageArrayGetter):
    def __init__(cls, resizer: callable, camera: any):
        if not resizer:
            raise ValueError('The resizer cannot be empty')
            
        if not camera:
            raise ValueError('The camera cannot be empty')
            
        super().__init__(camera, resizer)
        cls._camera = camera
        cls._resizer = resizer

    def __call__(cls, image_size: Tuple[int, int]) -> np.array:
        ret, image = cls._camera.read()
        if not ret:
            raise RuntimeError("Cannot get mage from the camera")

        image = cls._resizer(image, image_size)
        return image


class FileImageArrayGetter(ImageArrayGetter):
    def __init__(cls, resizer: callable):
        if not resizer:
            raise ValueError('The resizer cannot be empty')

        super().__init__(resizer)
        cls._resizer = resizer

    def __call__(cls, filename: str, image_size: Tuple[int, int]) -> np.array:
        image = cv2.imread(filename)
        image = cls._resizer(image, image_size)

        return image
