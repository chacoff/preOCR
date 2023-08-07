import cv2
import numpy as np
from math import atan2, pi


class CaoTools:
    """ several image processing helpers to read, crop and display images for çao """
    def __init__(self):
        super().__init__()
        self.image: np.array = None  # raw image
        self.roi: np.array = None  # image after roi
        self.channels: int = 0  # number of channels
        self.width: int = 0  # width of the raw image
        self.height: int = 0  # height of the raw image
        self.height_roi: int = 0  # height of the roi image
        self.width_roi: int = 0  # width of the roi image
        self.r: np.array = None  # red channel
        self.g: np.array = None  # green channel
        self.b: np.array = None  # blue channel

    def read_image(self, filename: str) -> any:
        """ read image and split it in all channels
        input = string with the image address
        return = self, image, height, width, channels, red, green and blue channel
        """

        self.image = cv2.imread(filename)  # x3
        self.height, self.width, self.channels = self.image.shape
        self.r, self.g, self.b = cv2.split(self.image)  # x3 (x1)
        # print(f'height={self.height}, width={self.width}, channels={self.channels}')

        return self

    def roi_image(self, channel: str, x: int = 20, y: int = 310) -> any:
        """ select a region of interest according the selected channel, it works best in the green channel
        input = image in Mat formatz
        parameters = channel selection to apply ROI and x, y coordinate to start the ROI
        return self and roi image
        """
        if channel == 'r':  # red channel
            img = self.r
        elif channel == 'g':  # green channel
            img = self.g
        elif channel == 'b':  # blue channel
            img = self.b
        else:
            img = self.g  # default is green channel

        self.roi = img[y:y + self.height, x:x + self.width]
        self.height_roi, self.width_roi = self.roi.shape

        return self

    @staticmethod
    def show_image_instack(numpy_stack: tuple, factor: float, direction: str) -> None:
        """ show image in a stack
            input example of a numpy stack = (self.r, self.g, self.b)
            direction = horizontal or vertical
            factor = the percentage of display
        """
        title = 'Image Stack'

        if direction == 'horizontal':
            cv2.imshow(title, cv2.resize(np.hstack(numpy_stack), None, fx=factor, fy=factor))
        elif direction == 'vertical':
            cv2.imshow(title, cv2.resize(np.vstack(numpy_stack), None, fx=factor, fy=factor))
        else:
            cv2.imshow(title, cv2.resize(np.hstack(numpy_stack), None, fx=factor, fy=factor))  # default is horizontal

        cv2.waitKey()

    @staticmethod
    def thresholding(img: np.array) -> np.array:
        """ thresholding the image for better contrast """
        image_thres, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        return img_bin

    @staticmethod
    def draw_grid(img: np.array, grid_shape: tuple, color: tuple = (0, 255, 0), thickness: int = 1):
        """ draw a grid in the image to estimate the sliding window """
        if len(img.shape) == 3:
            h, w, _ = img.shape
        else:
            h, w = img.shape

        rows, cols = grid_shape
        dy, dx = h / rows, w / cols

        # draw vertical lines
        for x in np.linspace(start=dx, stop=w - dx, num=cols - 1):
            x = int(round(x))
            cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

        # draw horizontal lines
        for y in np.linspace(start=dy, stop=h - dy, num=rows - 1):
            y = int(round(y))
            cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

        return img

    @staticmethod
    def resize_im(image: np.array, scale_percent: float) -> np.array:
        """ single resize with a factor of any input image """
        width = int(image.shape[1] * scale_percent)
        height = int(image.shape[0] * scale_percent)
        dim = (width, height)
        image_resized = cv2.resize(image, dim)

        return image_resized

    @staticmethod
    def coordinates_onclick(image: np.array) -> None:
        """ just to know the coordinate of the mouse to create a mask according the input image """

        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f'({x},{y})')
                cv2.putText(image, f'({x},{y})', (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(image, (x, y), 3, (0, 255, 255), -1)

        cv2.namedWindow('Point Coordinates')
        cv2.setMouseCallback('Point Coordinates', click_event)

        while True:
            cv2.imshow('Point Coordinates', image)
            q = cv2.waitKey(1) & 0xFF
            if q == 1:
                break
        cv2.destroyAllWindows()
