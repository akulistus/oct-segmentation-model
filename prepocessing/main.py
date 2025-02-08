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
    # equalized_image = cv2.equalizeHist(converted_image)
    # test1 = prep.fill_whites_with_noise(converted_image)
    filtered_image = prep.smooth_image(converted_image)
    sobel_image = prep.sobel(filtered_image)
    ret2, binary = prep.threshold(sobel_image)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = prep.morph_open(binary, element)
    binary = prep.morph_close(binary, element)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    approved_cnts = []
    for cnt in contours:
        contourArea = cv2.contourArea(cnt)
        if (contourArea > 250):
            x, y, w, h = cv2.boundingRect(cnt)
            if (w < image.shape[1] / 4):
                approved_cnts.append(cnt)
            elif (not prep.is_line(cnt)) :
                approved_cnts.append(cnt)
    
    x, y, w, h = cv2.boundingRect(np.vstack(approved_cnts))
    cv2.rectangle(image, (x, y), (x + w, y + h + 20), (0, 255, 0), 2)

    cv2.imwrite(f"result/{file}", image)
# # cv2.imshow("Denoised Image", image)
# # cv2.waitKey(0)
# cv2.destroyAllWindows()