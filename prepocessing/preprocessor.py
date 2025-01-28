import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.restoration import estimate_sigma
from bm3d import bm3d, BM3DStages

class Preprocessor:
    def __init__(self):
        pass

    def prepare(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        normalized_image = img_as_float(gray_image)
        denoised_image = self.remove_noise(normalized_image)
        edges = self.detect_edges(denoised_image)
        return edges

    def remove_noise(self, image):
        sigma = np.mean(estimate_sigma(image, average_sigmas=True)) * 10
        denoised_image = bm3d(image, sigma_psd=sigma, stage_arg=BM3DStages.ALL_STAGES)
        return denoised_image

    def detect_edges(self, image):
        image = (image * 255).astype('uint8')
        return cv2.Canny(image, 50, 100, L2gradient=False)
    
    def resize_image(self, image):
        pass

    def smooth_image(self, image):
        image_medianBlur = cv2.medianBlur(image, 17)
        image_gauss = cv2.GaussianBlur(image_medianBlur, (5,5),0)
        return image_gauss
    
    def convert_to_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def enhance_sharpness(self, image):
        return cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=5)

    def sobel(self, image):
        return cv2.Laplacian(image, -1, ksize=7)

    def threshold(self, image):
        return cv2.threshold(image, 200, 255, cv2.THRESH_OTSU)

    def get_bounding_box(self, image):
        pass