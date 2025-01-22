import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.restoration import estimate_sigma
from bm3d import bm3d, BM3DStages

test_image = "./data/test_image2.jpeg"
image = cv2.imread(test_image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
normalized_image = img_as_float(gray_image)

sigma_psd = np.mean(estimate_sigma(normalized_image)) * 10
denoised_image = bm3d(normalized_image, sigma_psd=sigma_psd, stage_arg=BM3DStages.ALL_STAGES)

denoised_image = (denoised_image * 255).astype('uint8')

cv2.imshow("Denoised Image", denoised_image)
# cv2.imwrite("test.jpeg", denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
