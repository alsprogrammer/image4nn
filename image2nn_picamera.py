from image2nn import ImageArrayGetter

from picamera import PiCamera
from picamera.array import PiRGBArray
from time import sleep


class RaspiCamImageArrayGetter(ImageArrayGetter):
    def __init__(cls):
    def __init__(cls, resizer: callable):
        if not resizer:
            raise ValueError('The resizer cannot be empty')
            
        super().__init__(resizer)
        cls._resizer = resizer        
        cls._camera = PiCamera()
        cls._rawCapture = PiRGBArray(cls._camera)
        sleep(0.1)

    def __call__(cls, image_size: Tuple[int, int]) -> np.array:
        cls._camera.capture(cls._rawCapture, format="bgr")
        image = rawCapture.array
        image = cls._resizer(image, image_size)

        return image
