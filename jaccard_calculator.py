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
            image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
            norm_image = cv2.normalize(image, None, alpha=0, beta=1,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            self.manual_images.append(norm_image.round())

    def calculate_coefficients(self):
        for dataset_name, images in self.output_data.items():
            self.jaccard_scores[dataset_name] = []
            for manual_image, processed_image in zip(self.manual_images, images):
                score = jaccard_similarity_score(manual_image.ravel(), processed_image[1].ravel().round())
                self.jaccard_scores[dataset_name].append(score)

        print(self.jaccard_scores)



