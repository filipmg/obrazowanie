from RidgeDetection.ridge_detector import RidgeDetector, DetectionType
import glob
import cv2
import os


class ImageProcessor:

    def __init__(self):
        self.ridgeDetector = RidgeDetector()
        self.file_list = None
        self.training_data = []

        self.output_data = {"ridge-opencv": [],
                            "ridge-custom": []}

        self.out_dirs = {"ridge-opencv": "DRIVE/processed/ridge-detection-opencv/",
                         "ridge-custom": "DRIVE/processed/ridge-detection-custom/"}

        self.create_output_dirs()

    def create_output_dirs(self):
        for directory in self.out_dirs.values():
            if not os.path.exists(directory):
                os.makedirs(directory)

    def load_data(self):
        self.file_list = glob.glob('DRIVE/training/images/*.tif')
        for filename in self.file_list:
            self.training_data.append(cv2.imread(filename))

    def save_data(self):
        idx = 0
        for dataset, images in self.output_data.items():
            for image in images:
                cv2.imwrite(self.out_dirs[dataset] + 'processed-' + str(idx) + '.tif', image)
                idx = idx + 1

    def process_data(self):
        for image in self.training_data:
            out_ridge_opencv = self.ridgeDetector.detect_ridges(image, DetectionType.OPENCV)
            out_ridge_custom = self.ridgeDetector.detect_ridges(image, DetectionType.CUSTOM)
            self.output_data["ridge-opencv"].append(out_ridge_opencv)
            self.output_data["ridge-custom"].append(out_ridge_custom)

    def get_processed_data(self):
        return self.output_data

    def start(self):
        self.load_data()
        self.process_data()
        self.save_data()
