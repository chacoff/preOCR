import cv2
import numpy as np
from cao_2 import CaoTools

cao = CaoTools()
cao.read_image('cao2.png').roi_image('g', x=20, y=310)
image = cao.roi.copy()  # green channel image

# find contours
# edged = cv2.Canny(image, 30, 180)
ret, edged = cv2.threshold(image, 100, 200, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# full = cv2.merge((image, image, image))
# full_drawn = full.copy()
# cv2.drawContours(full_drawn, contours, -1, (0, 255, 0), cv2.FILLED)
# full_drawn_resized = cao.resize_im(full_drawn, scale_percent=0.8)
# cv2.imshow('', full_drawn_resized)
# cv2.waitKey()

canvas = np.zeros((cao.height_roi, cao.width_roi, 3))
# cv2.fillPoly(canvas, pts=contours, color=(255, 255, 0))

for cnt in contours:
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    x, y, w, h = cv2.boundingRect(approx)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 1)

canvas = cao.resize_im(canvas, scale_percent=0.8)
cv2.imshow('', canvas)
cv2.waitKey()

# cao.show_image_instack((full_drawn, img_pl), factor=0.41, direction='horizontal')
print("Number of Contours found = " + str(len(contours)))
