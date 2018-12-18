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
            self.training_data.append((filename.rsplit('/', 1)[-1].rsplit('.')[0], cv2.imread(filename)))

    def save_data(self):
        for dataset, images in self.output_data.items():
            for image in images:
                cv2.imwrite(self.out_dirs[dataset] + image[0] + '-processed' + '.tif', image[1])

    def process_data(self):
        for image in self.training_data:
            image = (image[0], cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY))
            out_ridge_opencv = self.ridgeDetector.detect_ridges(image[1], DetectionType.OPENCV)
            out_ridge_custom = self.ridgeDetector.detect_ridges(image[1], DetectionType.CUSTOM)
            norm_image_out_ridge_opencv = cv2.normalize(out_ridge_opencv, None, alpha=0, beta=1,
                                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            norm_image_out_ridge_custom = cv2.normalize(out_ridge_custom, None, alpha=0, beta=1,
                                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            self.output_data["ridge-opencv"].append((image[0], norm_image_out_ridge_opencv.round()))
            self.output_data["ridge-custom"].append((image[0], norm_image_out_ridge_custom.round()))

    def get_processed_data(self):
        return self.output_data

    def start(self):
        self.load_data()
        self.process_data()
        self.save_data()
