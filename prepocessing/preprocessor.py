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
        sigma = np.mean(estimate_sigma(image, average_sigmas=True, channel_axis=-1)) * 10
        denoised_image = bm3d(image, sigma_psd=sigma, stage_arg=BM3DStages.ALL_STAGES)
        return denoised_image
    
    def resize_image(self, image, contours):
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        cropped = image[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (200, 200))
        return resized

    def get_contours(self, image):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        approved_cnts = []
        for cnt in contours:
            contourArea = cv2.contourArea(cnt)
            if (contourArea > 250):
                x, y, w, h = cv2.boundingRect(cnt)
                if (w < image.shape[1] / 4):
                    approved_cnts.append(cnt)
                elif (not self.is_line(cnt)) :
                    approved_cnts.append(cnt)
        return approved_cnts
    
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
        image_medianBlur = cv2.bilateralFilter(image_medianBlur, 11, 150, 150)
        return image_medianBlur
    
    def convert_to_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def sobel(self, image):
        return cv2.Sobel(image, -1, 0, 1, ksize=5)

    def threshold(self, image):
        return cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    def morph_open(self, image, element):
        binary = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        return binary
    
    def morph_close(self, image, element):
        binary = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element)
        return binary
        
    def find_contours(self, image):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy