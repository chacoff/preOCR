import cv2
import numpy as np
from mask import MaskProcessor
from text_finder import TextFinder


def resize_im(image: np.array, scale_percent: float) -> np.array:
    """ single resize with a factor of any input image """
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    dim = (width, height)
    image_resized = cv2.resize(image, dim)

    return image_resized


def roi_image(image: np.array, x: int = 20, y: int = 310) -> np.array:
    if len(image.shape) != 2:
        print('Image has to be 8bit')
        return

    height, width = image.shape
    roi = image[y:y + height, x:x + width]
    return roi


def main(image_path: str):
    image, mask = MaskProcessor(image_path).mask_extraction('g',
                                                            erode_vert=1,
                                                            dilate_vert=2,
                                                            erode_hori=1,
                                                            dilate_hori=2,
                                                            erode_weight=2,
                                                            weight=0.5,
                                                            kernel_rect=(3, 3),
                                                            debug=False)
    res = cv2.subtract(mask, image)
    res = 255 - res
    res = roi_image(res)
    res = resize_im(res, scale_percent=0.8)

    finder = TextFinder()
    finder.start(res, debug=False)

    finder.filter_boxes(min_area=40,
                        max_area=1000,
                        min_width=4,
                        max_width=9999,
                        min_height=4,
                        max_height=25)

    finder.merging_boxes(merge_x=26,
                         merge_y=5)

    finder.show_final_boxes()


if __name__ == '__main__':
    main('cao1.png')
