import cv2
from preprocessor import Preprocessor

test_image = "./data/test_image.jpeg"
image = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)

filtered_image = cv2.bilateralFilter(image, 21,51,51)
cv2.imshow("test", filtered_image)
cv2.waitKey(0)
