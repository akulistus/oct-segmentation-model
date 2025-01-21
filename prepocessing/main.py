import cv2
from bm3d import bm3d, BM3DStages

# Load and preprocess the image
test_image = "./data/test_image.jpeg"
image = cv2.imread(test_image)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Normalize pixel values to [0, 1]
normalized_image = gray_image / 255.0

# Apply BM3D denoising
sigma_psd = 50 / 255.0  # Adjust sigma_psd for normalized scale
denoised_image = bm3d(normalized_image, sigma_psd=sigma_psd, stage_arg=BM3DStages.ALL_STAGES)

# Denormalize back to [0, 255] and convert to uint8
denoised_image = (denoised_image * 255).astype('uint8')

# Display the result
cv2.imshow("Denoised Image", denoised_image)
# cv2.imwrite("test.jpeg", denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
