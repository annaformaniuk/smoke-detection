import numpy as np
import cv2 as cv
import imutils
from typing import List, Set, Dict, Tuple, Optional, Any
np.seterr(divide='ignore', invalid='ignore')


# to simply segment all the gray pixels
def simpleGray(bgr):
    # Saturation plus Value
    mask_white = np.ones(bgr.shape[:2], dtype="uint8")
    value = bgr.max(axis=2)
    dif = value-bgr.min(axis=2)
    saturation = np.nan_to_num(dif/value)
    mask_white[:, :] = ((value > 220) & (saturation < 0.20))*255
    cv.imwrite("saturationPlusValue.jpg", mask_white)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(mask_white, cv.MORPH_OPEN, kernel)
    cv.imwrite("02_opening_full.jpg", opening) # visualization
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    cv.imwrite("03_closing_full.jpg", closing) # visualization

    result_white = cv.bitwise_and(bgr, bgr, mask=closing)
    cnts = cv.findContours(closing.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cv.imwrite("04_result_white_full.jpg", result_white)
    print("Found {} possible smoke clouds in the original image".format(len(cnts)))
    # for latter search for intersections
    better_format = []
    for c in cnts:
        single_contour = []
        for point in c:
            single_contour.append((point[0,0], point[0,1]))

        better_format.append(single_contour)

    return cnts, better_format

# determine if a point is inside a given polygon or not
# Polygon is a list of (x,y) pairs.
def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


cap = cv.VideoCapture('features/images/YUNC0025_Trim.mp4')
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6))
fgbg = cv.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False)
while(1):
    ret, frame = cap.read()
    # counting what frame it is
    i = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    # stop before the video ends not to give an error
    if frame is None:
        break

    if (i > 30)&(i % 10 == 0):      

        # applying the bs
        fgmask = fgbg.apply(frame)
        # erosion and dilation
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

        # showing (or not showing the result of foreground extraction)
        # cv.imshow('frame', result_white)

        #stopping at frame number...
        if i == 300:
            # selecting the color pixels from the foreground
            cv.imwrite("01_fullframe.jpg", frame)  # visualization
            color = cv.bitwise_and(frame, frame, mask=fgmask)
            cv.imwrite("05_color_foreground.jpg", color)  # visualization

            # Saturation Plus Value
            mask_white = np.ones(color.shape[:2], dtype="uint8")
            value = color.max(axis=2)
            dif = value - color.min(axis=2)
            saturation = np.nan_to_num(dif/value)
            mask_white[:, :] = ((value > 220) & (saturation < 0.20))*255

            cv.imwrite("06_mask_white_foreground.jpg", mask_white)  # visualization
            # the pixels
            result_white = cv.bitwise_and(color, color, mask=mask_white)
            cv.imwrite("07_result_white_foreground.jpg", result_white)  # visualization

            # opening and closing
            kernel = np.ones((5, 5), np.uint8)
            closing = cv.morphologyEx(mask_white, cv.MORPH_CLOSE, kernel)

            # grabbing the contours
            cnts = cv.findContours(closing.copy(), cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            print("Found {} possible smoke clouds in the foreground".format(len(cnts)))
            # cv.imshow("Mask", closing)

            # now trying to see whether some of the grey regions in the frame are also moving
            original_grey, color_seg_cnts = simpleGray(frame)
            overlapping_contours: List[int] = []
            index = 0

            # loop over the contours from colour segmentation
            for whole_cnt in color_seg_cnts:
                # loop over the contours detected after foreground extraction
                for c in cnts:
                    # counting how many are in
                    results = [] # type: List[bool]

                    # loop over each point in each contour object
                    for single in c:
                        inside = point_inside_polygon(single[0,0], single[0,1], whole_cnt)
                        results.append(inside)
                        # print(inside)

                    positives = sum(x == True for x in results)
                    # appending the grey objects to the final result
                    if (positives > len(c)/1.5):
                        # for arr in overlapping_contours:
                        if not any(np.array_equal(original_grey[index], arr) for arr in overlapping_contours):
                            overlapping_contours.append(original_grey[index])

                index +=1

            print("Resulting smoke clouds")
            print(len(overlapping_contours))
            # draw the contour and show it
            for c in overlapping_contours:
                area = cv.contourArea(c)
                print(area)
                print(cv.isContourConvex(c))
                cv.drawContours(frame, [c], -1, (0, 255, 0), 2)
                cv.imshow("Image", frame)
                cv.waitKey(0)

        # k = cv.waitKey(30) & 0xff
        # if k == 27:
        #         break
cap.release()
cv.destroyAllWindows()