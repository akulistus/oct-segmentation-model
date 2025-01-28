import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("./data/test_image3.jpeg")  # Replace with your image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Define color space conversions
color_spaces = {
    "RGB": image_rgb,
    "Grayscale": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
    "HSV": cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
    "Lab": cv2.cvtColor(image, cv2.COLOR_BGR2LAB),
    "Luv": cv2.cvtColor(image, cv2.COLOR_BGR2LUV),
    "XYZ": cv2.cvtColor(image, cv2.COLOR_BGR2XYZ),
    "YCrCb": cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb),
    "HLS": cv2.cvtColor(image, cv2.COLOR_BGR2HLS),
    "YUV": cv2.cvtColor(image, cv2.COLOR_BGR2YUV),
}

# Plot images in different color spaces
num_spaces = len(color_spaces)
fig, axs = plt.subplots(1, num_spaces, figsize=(20, 5))

for i, (space_name, img) in enumerate(color_spaces.items()):
    if len(img.shape) == 2:  # Grayscale image
        axs[i].imshow(img, cmap="gray")
    else:  # Color image
        if space_name != "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert for proper display
        axs[i].imshow(img)
    
    axs[i].set_title(space_name)
    axs[i].axis("off")

plt.tight_layout()
plt.show()
