import cv2
import numpy as np


class MaskProcessor:
    """ it generates a mask based on vertical and horizontal lines in order to isolate the text and remove drawings """

    def __init__(self, image: str = 'image.png'):
        super().__init__()
        self.input_image: np.array = cv2.imread(image)
        self.r, self.g, self.b = cv2.split(self.input_image)
        self.channel_copy: np.array = self.g.copy()

    def channel_selector(self, channel: str) -> np.array:
        """ channel selector for processing """
        # TODO: because looks ugly

        if channel == 'r':
            img_channel = self.r
        elif channel == 'g':
            img_channel = self.g
        elif channel == 'b':
            img_channel = self.b
        else:
            img_channel = self.g

        return img_channel

    def mask_extraction(self,
                        channel: str,
                        weight: float = 0.5,
                        erode_vert: int = 1,
                        dilate_vert: int = 2,
                        erode_hori: int = 2,
                        dilate_hori: int = 2,
                        erode_weight: int = 2,
                        kernel_rect: tuple = (3, 3),
                        debug: bool = False) -> (np.array, np.array):

        """ thresholding and inversion of the image for box extraction
        Parameters:
            alpha = weight the image
            debug = to write on out pre processed images

        Explanation:
        kernel_length = numpy array of the length of the image
        vertical_kernel = verticle kernel (1 X kernel_length), it will detect all the verticle lines
        horizontal_kernal = horizontal kernel (kernel_length X 1), it  will  detect all the horizontal lines
        """

        img_channel = self.channel_selector(channel)

        (thresh, binary_image) = cv2.threshold(img_channel, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary_image = 255 - binary_image

        if debug:
            cv2.imwrite("debug/binary_image.jpg", binary_image)

        kernel_length = np.array(img_channel).shape[1] // 40  # integer division

        # Vertical morpho operations
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        img_temp1 = cv2.erode(binary_image, vertical_kernel, iterations=erode_vert)
        vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=dilate_vert)

        if debug:
            cv2.imwrite("debug/vertical_lines.jpg", vertical_lines_img)

        # Horizontal morpho operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        img_temp2 = cv2.erode(binary_image, horizontal_kernel, iterations=erode_hori)
        horizontal_lines_img = cv2.dilate(img_temp2, horizontal_kernel, iterations=dilate_hori)

        if debug:
            cv2.imwrite("debug/horizontal_lines.jpg", horizontal_lines_img)

        # Weighting parameters, this will decide the quantity of an image to be added to make a new image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_rect)
        final_mask = cv2.addWeighted(vertical_lines_img, weight, horizontal_lines_img, (1 - weight), 0.0)
        final_mask = cv2.erode(~final_mask, kernel, iterations=erode_weight)
        (thresh, final_mask) = cv2.threshold(final_mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if debug:
            cv2.imwrite("debug/final_binary_mask.jpg", final_mask)

        return self.channel_copy, final_mask


