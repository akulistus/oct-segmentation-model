import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessor import Preprocessor

test_image = "./data/test_image9.jpeg"
image = cv2.imread(test_image)

prep = Preprocessor()

converted_image = prep.convert_to_gray(image)
filtered_image = prep.smooth_image(converted_image)
sobel_image = prep.sobel(filtered_image)
# sharpend_image = prep.enhance_sharpness(filtered_image)
ret2, binary = prep.threshold(sobel_image)
# plt.hist(filtered_image.ravel(), bins=256, range=(0, 256))
# plt.show()

cv2.imshow("Denoised Image", binary)
# # cv2.imwrite("test.jpeg", denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
