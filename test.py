import numpy as np
import cv2 as cv
import imutils


cap = cv.VideoCapture('features/images/short.mp4')
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

    # applying the bs
    fgmask = fgbg.apply(frame)
    # erosion and dilation
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

    # showing (or not showing the result)
    # cv.imshow('frame', result_white)

    if i == 100:
        # selecting the color pixels from the foreground
        color = cv.bitwise_and(frame, frame, mask=fgmask)
        cv.imwrite("color.jpg", color) # visualization

        hsv_image = cv.cvtColor(color, cv.COLOR_BGR2HSV)
        cv.imwrite("hsv_image.jpg", hsv_image) # visualization pretty in pink
        # range
        light_white = (0, 0, 200)
        dark_white = (145, 60, 255)
        # the mask
        mask_white = cv.inRange(hsv_image, light_white, dark_white)
        cv.imwrite("mask_white.jpg", mask_white) # visualization
        # the pixels
        result_white = cv.bitwise_and(color, color, mask=mask_white)
        cv.imwrite("result_white.jpg", result_white) # visualization
        # find the contours in the mask
        cnts = cv.findContours(mask_white.copy(), cv.RETR_EXTERNAL,
                       cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        print("Found {} possible smoke clouds".format(len(cnts)))
        cv.imshow("Mask", mask_white)

        # loop over the contours
        for c in cnts:
            # draw the contour and show it
            cv.drawContours(result_white, [c], -1, (0, 255, 0), 2)
            cv.imshow("Image", result_white)
            cv.waitKey(0)

    # k = cv.waitKey(30) & 0xff
    # if k == 27:
    #         break
cap.release()
cv.destroyAllWindows()