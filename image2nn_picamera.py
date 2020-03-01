from image2nn import ImageArrayGetter

from picamera import PiCamera
from picamera.array import PiRGBArray
from time import sleep


class RaspiCamImageArrayGetter(ImageArrayGetter):
    def __init__(cls):
        cls._camera = PiCamera()
        cls._rawCapture = PiRGBArray(cls._camera)
        sleep(0.1)

    def __call__(cls) -> np.array:
        cls._camera.capture(rawCapture, format="bgr")
        image = rawCapture.array

        return image
