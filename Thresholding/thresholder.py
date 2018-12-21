from skimage.filters import threshold_local
from enum import Enum
import cv2

class AdaptiveThreshType(Enum):
    GAUSSIAN = 1
    MEAN = 2

class Thresholder:

    def thresh(self, image, adaptiveThreshType):
        if adaptiveThreshType == AdaptiveThreshType.MEAN:
            out_image = self.__detect_using_mean(image)
        elif adaptiveThreshType == AdaptiveThreshType.GAUSSIAN:
            out_image = self.__detect_using_gaussian(image)
        else:
            out_image = self.__detect_using_gaussian(image)

        return out_image

    def __detect_using_mean(self, image):
        binary_local = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return binary_local

    def __detect_using_gaussian(self, image):
        binary_local = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return binary_local
        