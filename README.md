# image4nn

A small library for geting, managing and converting images for artificial neural networks (Keras and PyTorch are supported).

It allows to get images from camera in linux, from a file or feom a Raspberry Pi camera.

The image can be resized, and then converted to a PyTorch tensor.

For using just clone the repository, copy the files to your project dir, and use it.

## Examples

### From usb camera

```
import cv2
from image4nn import CV2CamImageArrayGetter


image_size = (320, 240)
camera = cv2.VideoCapture(0)
get_image = CV2CamImageArrayGetter(cv2.resize, camera)

image = get_image(image_size)

print(image)
```

### From usb camera to PyTorch tensor

```
import torch
import cv2

from image4nn import CV2CamImageArrayGetter
from image2nn_torch import NPArrayToTensor

image_size = (320, 240)
camera = cv2.VideoCapture(0)

converter2tensor = NPArrayToTensor()

get_image = CV2CamImageArrayGetter(cv2.resize, camera)

image = get_image(image_size)

image_tensor = converter2tensor(image)
print(image_tensor)
```
