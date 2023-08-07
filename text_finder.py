import cv2
import numpy as np


class TextFinder:
    def __init__(self):
        super().__init__()
        self.img: np.array = None
        self.copy_img: np.array = None
        self.copy_img_for_debug: np.array = None
        self.copy_for_results: np.array = None
        self.boxes = []
        self.debug = False

    def start(self, image: np.array, debug: bool = False):
        """ starts the process by updating:
                self.image
                self.orig """
        self.debug = debug
        self.get_input_image(image)  # updates self.img and self.orig
        self.median_canny(0, 1)  # updates self.img
        self.contours_and_rectangles()  # calculates all contours and return a list with boxes

    def get_input_image(self, image: np.array) -> None:
        self.img = image
        self.copy_img = np.copy(image)
        self.copy_img_for_debug = np.copy(image)
        self.copy_for_results = np.copy(image)

    def median_canny(self, thresh1: float, thresh2: float) -> None:
        median = np.median(self.img)
        self.img = cv2.Canny(self.img, int(thresh1 * median), int(thresh2 * median))
        if self.debug:
            cv2.imwrite('debug/finder_canny_median.png', self.img)

    @staticmethod
    def tup(point: list) -> tuple:
        return point[0], point[1]

    @staticmethod
    def overlap(source, target) -> bool:
        """ true if 2 boxes overlap """
        # unpack points
        tl1, br1 = source
        tl2, br2 = target

        # checks
        if tl1[0] >= br2[0] or tl2[0] >= br1[0]:
            return False
        if tl1[1] >= br2[1] or tl2[1] >= br1[1]:
            return False
        return True

    def get_all_overlaps(self, boxes: list, bounds, index) -> list:
        """ returns all overlapping boxes """
        overlaps = []
        for a in range(len(boxes)):
            if a != index:
                if self.overlap(bounds, boxes[a]):
                    overlaps.append(a)
        return overlaps

    def contours_and_rectangles(self) -> None:
        """ find contours and go through the contours and save the box edges,
        draw the first rectangles before merging them """

        contours, hierarchy = cv2.findContours(self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # each element is [[top-left], [bottom-right]];
        hierarchy = hierarchy[0]
        for component in zip(contours, hierarchy):
            current_contour = component[0]
            current_hierarchy = component[1]
            x, y, w, h = cv2.boundingRect(current_contour)
            if current_hierarchy[3] < 0:
                cv2.rectangle(self.copy_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                self.boxes.append([[x, y], [x + w, y + h]])

        if self.debug:
            cv2.imwrite('debug/finder_contours_rectangles.png', self.copy_img)

    def filter_boxes(self,
                     min_area: int = 40,
                     max_area: int = 1000,
                     min_width: int = 4,
                     max_width: int = 9999,
                     min_height: int = 4,
                     max_height: int = 25) -> None:

        """ filter out excessively small and large boxes """
        filtered = []
        for box in self.boxes:
            w = box[1][0] - box[0][0]
            h = box[1][1] - box[0][1]
            if min_area < w * h < max_area and w > min_width and h > min_height:
                filtered.append(box)

        self.boxes = filtered

    def merging_boxes(self,
                      merge_x: int = 26,
                      merge_y: int = 10):
        """ go through the boxes and start merging """
        finished = False
        highlight = [[0, 0], [1, 1]]
        points = [[[0, 0]]]

        while not finished:
            # set end con
            finished = True

            if self.debug:
                for box in self.boxes:
                    cv2.rectangle(self.copy_img_for_debug, self.tup(box[0]), self.tup(box[1]), (0, 200, 0), 1)
                cv2.rectangle(self.copy_img_for_debug, self.tup(highlight[0]), self.tup(highlight[1]), (0, 0, 255), 2)
                for point in points:
                    point = point[0]
                    cv2.circle(self.copy_img_for_debug, self.tup(point), 4, (255, 0, 0), -1)
                cv2.imshow("Copy", self.copy_img_for_debug)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

            # loop through boxes
            index = len(self.boxes) - 1
            while index >= 0:
                # grab current box
                curr = self.boxes[index]

                # add margin
                tl = curr[0][:]
                br = curr[1][:]
                tl[0] -= merge_x
                tl[1] -= merge_y
                br[0] += merge_x
                br[1] += merge_y

                # get matching boxes
                overlaps = self.get_all_overlaps(self.boxes, [tl, br], index)

                # check if empty
                if len(overlaps) > 0:
                    # combine boxes
                    # convert to a contour
                    con = []
                    overlaps.append(index)
                    for ind in overlaps:
                        tl, br = self.boxes[ind]
                        con.append([tl])
                        con.append([br])
                    con = np.array(con)

                    # get bounding rect
                    x, y, w, h = cv2.boundingRect(con)

                    # stop growing
                    w -= 1
                    h -= 1
                    merged = [[x, y], [x + w, y + h]]

                    # highlights
                    highlight = merged[:]
                    points = con

                    # remove boxes from list
                    overlaps.sort(reverse=True)
                    for ind in overlaps:
                        del self.boxes[ind]
                    self.boxes.append(merged)

                    # set flag
                    finished = False
                    break

                # increment
                index -= 1
        cv2.destroyAllWindows()

    def show_final_boxes(self):
        """ show the end result """
        for box in self.boxes:
            cv2.rectangle(self.copy_for_results, self.tup(box[0]), self.tup(box[1]), (0, 200, 0), 1)
        cv2.imshow("Final", self.copy_for_results)
        cv2.waitKey(0)


def main():
    img = cv2.imread("clean.png")

    finder = TextFinder()
    finder.start(img, debug=True)

    finder.filter_boxes(min_area=40,
                        max_area=1000,
                        min_width=4,
                        max_width=9999,
                        min_height=4,
                        max_height=25)

    finder.merging_boxes(merge_x=26,
                         merge_y=10)

    finder.show_final_boxes()


if __name__ == '__main__':
    main()
