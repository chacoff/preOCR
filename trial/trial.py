from skimage.feature import hog
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class SlideHog:
    image_height = 48
    image_width = 48
    window_size = 24
    window_step = 6

    @staticmethod
    def sliding_hog_windows(image: np.array, height: int, width: int, step: int = 280, window_size: int = 300):
        hog_vector = []
        for y in range(0, height, step):
            for x in range(0, width, step):
                window = image[y:y + window_size, x:x + window_size]
                hog_vector.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                      cells_per_block=(1, 1), visualise=False))
        return hog_vector


def main():
    slide = SlideHog()
    # image = np.load('image.npy')
    image = cv2.imread('../cao2.png')
    hog_vector = slide.sliding_hog_windows(image, image.shape[0], image.shape[1])


if __name__ == '__main__':
    main()