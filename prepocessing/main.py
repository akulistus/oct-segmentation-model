import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessor import Preprocessor

dir = "./experiments"
image = cv2.imread("./data/test_image9.jpeg")
prep = Preprocessor()
converted_image = prep.convert_to_gray(image)
test2 = cv2.equalizeHist(converted_image)
filtered_image = prep.smooth_image(test2)
sobel_image = prep.sobel(filtered_image)
ret2, binary = prep.threshold(sobel_image)
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
binary = prep.morph_open(binary, element)
binary = prep.morph_close(binary, element)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
for cnt in contours:
    contourArea = cv2.contourArea(cnt)
    if (contourArea > 2000):
        cv2.drawContours(image, [cnt], -1, (255, 255, 0), 2)
cv2.imshow("Denoised Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()