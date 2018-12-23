from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import numpy as np
import cv2

class RidgeDetector:

    def detect_ridges(self, image, detection_type):
        if detection_type == 1:
            out_image = self.__detect_using_custom(image)
        elif detection_type == 2:
            out_image = self.__detect_using_opencv(image)
        else:
            out_image = self.__detect_using_opencv(image)

        return out_image

    def __detect_using_custom(self, image):
        h_elems = hessian_matrix(image=np.float32(image), sigma=1.5, order='xy')
        eigvals = hessian_matrix_eigvals(h_elems)
        return eigvals[0]

    def __detect_using_opencv(self, image):
        ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
        ridges = ridge_filter.getRidgeFilteredImage(image)
        return ridges
