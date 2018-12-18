from image_processor import ImageProcessor
from jaccard_calculator import JaccardCalculator

if __name__ == "__main__":
    imageProcessor = ImageProcessor()
    imageProcessor.start()

    jaccardCalculator = JaccardCalculator(imageProcessor.get_processed_data())
    #jaccardCalculator.calculate_coefficients()
