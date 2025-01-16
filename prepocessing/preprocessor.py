import cv2
import numpy as np

class Preprocessor:
    def __init__(self):
        pass    
    def remove_pepper_salt_noise(image, max_window_size):
        if max_window_size % 2 == 0:
            raise ValueError("max_window_size must be odd")

        padded_image = np.pad(image, max_window_size // 2, mode='constant', constant_values=0)
        output_image = image.copy()

        rows, cols = image.shape

        for row in range(rows):
            for col in range(cols):
                window_size = 3

                while window_size <= max_window_size:
                    local_region = padded_image[row:row + window_size, col:col+window_size]

                    Z_min = np.min(local_region)
                    Z_max = np.max(local_region)
                    Z_med = np.median(local_region)
                    Z_xy = image[row, col]

                    A1 = Z_med - Z_min
                    A2 = Z_med - Z_max

                    if A1 > 0 and A2 < 0:
                        B1 = Z_xy - Z_min
                        B2 = Z_xy - Z_max
                        if B1 > 0 and B2 < 0:
                            output_image[row, col] = Z_xy
                        else:
                            output_image[row, col] = Z_med
                        break
                    else:
                        window_size += 2 

                if window_size > max_window_size:
                    output_image[row, col] = Z_med

        return output_image