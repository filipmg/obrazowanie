from RidgeDetection.ridge_detector import RidgeDetector, DetectionType
import glob
import cv2
import os


class ImageProcessor:

    def __init__(self):
        self.ridgeDetector = RidgeDetector()
        self.file_list = None
        self.training_data = []
        self.ridge_detection_output_data = []

        self.ridge_opencv_outdir = 'DRIVE/processed/ridge-detection-opencv/'

    def load_data(self):
        self.file_list = glob.glob('DRIVE/training/images/*.tif')
        for filename in self.file_list:
            self.training_data.append(cv2.imread(filename))

    def save_data(self):
        if not os.path.exists(self.ridge_opencv_outdir):
            os.makedirs(self.ridge_opencv_outdir)
        idx = 0
        for image in self.ridge_detection_output_data:
            cv2.imwrite(self.ridge_opencv_outdir + 'processed-' + str(idx) + '.tif', image)
            idx = idx + 1

    def process_data(self):
        for image in self.training_data:
            out_image = self.ridgeDetector.detect_ridges(image, DetectionType.OPENCV)
            self.ridge_detection_output_data.append(out_image)

    def start(self):
        self.load_data()
        self.process_data()
        self.save_data()
