from RidgeDetection.ridge_detector import RidgeDetector
from Thresholding.thresholder import Thresholder, DetectionType
import glob
import cv2
import os
import statistics
from PIL import Image
import numpy
import datetime


class ImageProcessor:

    def __init__(self):
        self.ridgeDetector = RidgeDetector()
        self.thresholder = Thresholder()
        self.file_list = None
        self.training_data = []
        self.training_data_mask = []

        self.output_data = {"ridge-opencv": [],
                            "ridge-custom": [],
                            "thresh-custom": [],
                            "thresh-mean": []}

        self.out_dirs = {"ridge-custom": "DRIVE/processed/ridge-detection-custom/",
                         "ridge-opencv": "DRIVE/processed/ridge-detection-opencv/",
                         "thresh-custom": "DRIVE/processed/thresholding-custom/",
                         "thresh-mean": "DRIVE/processed/thresholding-mean/"}

        self.create_output_dirs()

    def create_output_dirs(self):
        for directory in self.out_dirs.values():
            if not os.path.exists(directory):
                os.makedirs(directory)

    def load_data(self):
        self.load_data_images()
        self.load_data_masks()

    def load_data_images(self):
        self.file_list = glob.glob('DRIVE/training/images/*.tif')
        for filename in self.file_list:
            self.training_data.append((filename.rsplit('/', 1)[-1].rsplit('.')[0], cv2.imread(filename)))
        
    def load_data_masks(self):
        self.file_list = glob.glob('DRIVE/training/mask/*.gif')
        for filename in self.file_list:
            self.training_data_mask.append(numpy.asarray(Image.open(filename)))

    def save_data(self):
        for dataset, images in self.output_data.items():
            for image in images:
                cv2.imwrite(self.out_dirs[dataset] + image[0] + '-processed' + '.tif', image[1])

    def process_data(self):

        # Ridge detection
        ridge_thresholding = 5.5

        t1 = datetime.datetime.now()
        # Custom
        for image, mask in zip(self.training_data, self.training_data_mask):
            image = (image[0], cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY))
            out_ridge_custom = self.ridgeDetector.detect_ridges(image[1], 2)

            out_ridge_custom[out_ridge_custom > ridge_thresholding*statistics.median(out_ridge_custom.ravel())] = 255
            out_ridge_custom[out_ridge_custom <= ridge_thresholding*statistics.median(out_ridge_custom.ravel())] = 0
            self.output_data["ridge-custom"].append((image[0], out_ridge_custom))
        t2 = datetime.datetime.now()

        time = t2-t1
        print("Custom ridge detection time, seconds: ", time.seconds, " microseconds: ", time.microseconds)

        t1 = datetime.datetime.now()
        # OpenCV
        for image, mask in zip(self.training_data, self.training_data_mask):
            image = (image[0], cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY))
            out_ridge_opencv = self.ridgeDetector.detect_ridges(image[1], 1)

            out_ridge_opencv[out_ridge_opencv > ridge_thresholding*statistics.median(out_ridge_opencv.ravel())] = 255
            out_ridge_opencv[out_ridge_opencv <= ridge_thresholding*statistics.median(out_ridge_opencv.ravel())] = 0
            self.output_data["ridge-opencv"].append((image[0], out_ridge_opencv))
        t2 = datetime.datetime.now()

        time = t2-t1
        print("OpenCV ridge detection time, seconds: ", time.seconds, " microseconds: ", time.microseconds)

        # Thresholding
        self.proces_data_threasholding()
    
    def proces_data_threasholding(self):
        for image in self.training_data:
            image = (image[0], cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY))
            out_thresh_mean = self.thresholder.thresh(image[1], DetectionType.OPENCV)
            self.output_data["thresh-mean"].append((image[0], out_thresh_mean)) 

        for image in self.training_data:
            mask = self.training_data_mask[self.training_data.index(image)]
            out_thresh_custom = self.thresholder.thresh(image[1], DetectionType.CUSTOM, mask=mask)
            self.output_data["thresh-custom"].append((image[0], out_thresh_custom))
    

    def get_processed_data(self):
        return self.output_data

    def start(self):
        self.load_data()
        self.process_data()
        self.save_data()
