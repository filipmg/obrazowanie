from sklearn.metrics import jaccard_similarity_score, zero_one_loss
import glob
import cv2
import statistics


class JaccardCalculator:

    def __init__(self, output_data):
        self.manual_dir = "DRIVE\\training\\1st_manual\\*.png"
        self.unet_manual_dir = "UNet\\data\\vines\\test\\*predict.png"
        self.output_data = output_data
        self.manual_images = []
        self.jaccard_scores = {}
        self.zero_one_lose_scores = {}
        self.file_list = None

        self.load_manuals()

    def load_manuals(self):
        self.file_list = glob.glob(self.manual_dir)
        for filename in self.file_list:
            image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
            self.manual_images.append(image)
    
    def reload_manuals(self, manuals_directory):
        self.manual_images = []
        self.manual_dir = manuals_directory
        self.load_manuals()

    def calculate_jaccard_scores(self, dataset_name, images):
        for manual_image, processed_image in zip(self.manual_images, images):
            zero_one_loss_score = zero_one_loss(manual_image.ravel(), processed_image[1].ravel())
            score = jaccard_similarity_score(manual_image.ravel(), processed_image[1].ravel())
            self.jaccard_scores[dataset_name].append(score)
            self.zero_one_lose_scores[dataset_name].append(zero_one_loss_score)

    def calculate_coefficients_for_unet(self, dataset_name, images):
        self.reload_manuals(self.unet_manual_dir)
        self.calculate_jaccard_scores(dataset_name, images)
        self.reload_manuals(self.manual_dir)

    def calculate_coefficients(self):
        for dataset_name, images in self.output_data.items():
            self.jaccard_scores[dataset_name] = []
            self.zero_one_lose_scores[dataset_name] = []
            if(dataset_name == "unet"):
                self.calculate_coefficients_for_unet(dataset_name, images)
            else:
                self.calculate_jaccard_scores(dataset_name, images)

        print("Ridge custom, mean Jaccard score: ", statistics.mean(self.jaccard_scores["ridge-custom"]))
        print("Ridge custom, mean Zero One Loss score: ", statistics.mean(self.zero_one_lose_scores["ridge-custom"]))

        print("Ridge OpenCV, mean Jaccard score: ", statistics.mean(self.jaccard_scores["ridge-opencv"]))
        print("Ridge OpenCV, mean Zero One Loss score: ", statistics.mean(self.zero_one_lose_scores["ridge-opencv"]))

        print("Threshold custom, mean Jaccard score: ", statistics.mean(self.jaccard_scores["thresh-custom"]))
        print("Threshold custom, mean Zero One Loss score: ", statistics.mean(self.zero_one_lose_scores["thresh-custom"]))

        print("Threshold OpenCV, mean Jaccard score: ", statistics.mean(self.jaccard_scores["thresh-mean"]))
        print("Threshold OpenCV, mean Zero One Loss score: ", statistics.mean(self.zero_one_lose_scores["thresh-mean"]))

     #   print("UNet, mean Jaccard score: ", statistics.mean(self.jaccard_scores["unet"]))
     #   print("UNet, mean Zero One Loss score: ", statistics.mean(self.zero_one_lose_scores["unet"]))




