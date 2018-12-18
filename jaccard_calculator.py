from sklearn.metrics import jaccard_similarity_score
import glob
import cv2


class JaccardCalculator:

    def __init__(self, output_data):

        self.manual_dir = "DRIVE/training/1st_manual/*.png"
        self.output_data = output_data
        self.manual_images = []
        self.jaccard_scores = {}
        self.file_list = None

        self.load_manuals()

    def load_manuals(self):
        self.file_list = glob.glob(self.manual_dir)
        for filename in self.file_list:
            self.manual_images.append(cv2.imread(filename))

    def calculate_coefficients(self):
        for dataset_name, images in self.output_data.items():
            self.jaccard_scores[dataset_name] = []
            for manual_image, processed_image in zip(self.manual_images, images):
                score = jaccard_similarity_score(manual_image, processed_image)
                self.jaccard_scores[dataset_name].append(score)

        print(self.jaccard_scores)



