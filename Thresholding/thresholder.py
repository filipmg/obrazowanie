from skimage.filters import threshold_local
import cv2
from enum import Enum

class DetectionType(Enum):
    CUSTOM = 1
    OPENCV = 2

class Thresholder:
    
    def thresh(self, image, detection_type, mask=None):
        if detection_type == DetectionType.CUSTOM:
            out_image = self.__detect_using_custom(image, mask)
        elif detection_type == DetectionType.OPENCV:
            out_image = self.__detect_using_mean(image)
        else:
            out_image = self.__detect_using_mean(image)

        return out_image

    def __detect_using_mean(self, image):
        binary_local = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return binary_local

    def __detect_using_gaussian(self, image):
        binary_local = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return binary_local
        
    def __detect_using_custom(self, colorImage, mask):
        blueChan, greenChan, redChan = cv2.split(colorImage)
        
        histogramGreen = cv2.calcHist([greenChan], [0], mask, [256], [0, 256])

        threshValueGreen = self.calculateThreshValue(histogramGreen)

        self.threshAtValue(greenChan, threshValueGreen)

        return greenChan

    def calculateThreshValue(self, histogram):
        sum = 0
        for i in range(0, 256):
            sum += histogram[i][0]
            if sum >= 15000:
                return i


    def threshAtValue(self, image, treshValue):
        image[image >= treshValue] = 255
        image[image < treshValue] = 0
