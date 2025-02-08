import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from preprocessor import Preprocessor

dir = "./experiments"
# image = cv2.imread("./data/test_image10.jpeg")
prep = Preprocessor()
files = os.listdir(dir)
for file in files: 
    image = cv2.imread(f"{dir}/{file}")
    converted_image = prep.convert_to_gray(image)
    filtered_image = prep.smooth_image(converted_image)
    sobel_image = prep.sobel(filtered_image)
    ret2, binary = prep.threshold(sobel_image)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = prep.morph_open(binary, element)
    binary = prep.morph_close(binary, element)
    contours = prep.get_contours(binary)
    resized = prep.resize_image(image, contours)
    denoised_image = prep.remove_noise(resized)

    cv2.imwrite(f"result/{file}", denoised_image)
# # cv2.imshow("Denoised Image", image)
# # cv2.waitKey(0)
# cv2.destroyAllWindows()