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
    
    def resize_image(self, image):
        pass

    def fill_whites_with_noise(self, image):
        image_copy = image.copy()
        _, mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
        mask_blurred = cv2.medianBlur(mask, 101)
        mean_noise = estimate_sigma(image.astype(np.float32), average_sigmas=True) * 10
        noise = np.random.normal(mean_noise, 50, image.shape).astype(np.uint8)
        image_copy = (image + noise * (mask_blurred / 255.0)).astype(np.uint8) 
        inpainted = cv2.inpaint(image_copy, mask_blurred, inpaintRadius=150, flags=cv2.INPAINT_TELEA)
        return inpainted
    
    def is_line(self, contour):
        [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

        errors = []
        for point in contour:
            x, y = point[0]
            distance = abs(vy * (x - x0) - vx * (y - y0))
            errors.append(distance)
        return max(errors) < 13 # 12

    def smooth_image(self, image):
        image_medianBlur = cv2.medianBlur(image, 21)
        image_medianBlur = cv2.bilateralFilter(image_medianBlur, 11, 100, 100)
        return image_medianBlur
    
    def convert_to_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def sobel(self, image):
        return cv2.Sobel(image, -1, 0, 1, ksize=5)

    def threshold(self, image):
        return cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    
    def get_bounding_box(self, image):
        pass

    def morph_open(self, image, element):
        binary = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        return binary
    
    def morph_close(self, image, element):
        binary = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element)
        return binary
        
    def find_contours(self, image):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy