import numpy as np
import cv2 as cv


cap = cv.VideoCapture('features/images/DJI_0843.mp4')
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6))
fgbg = cv.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False)
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    color = cv.bitwise_and(frame, frame, mask=fgmask)
    cv.imwrite("color.jpg", color)

    hsv_image = cv.cvtColor(color, cv.COLOR_BGR2HSV)
    cv.imwrite("hsv_image.jpg", hsv_image)
    light_white = (0, 0, 200)
    dark_white = (145, 60, 255)
    mask_white = cv.inRange(hsv_image, light_white, dark_white)
    cv.imwrite("mask_white.jpg", mask_white)
    result_white = cv.bitwise_and(color, color, mask=mask_white)
    cv.imwrite("result_white.jpg", result_white)

    cv.imshow('frame', result_white)
    k = cv.waitKey(30) & 0xff
    if k == 27:
            break
cap.release()
cv.destroyAllWindows()


# is slower
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(6,6))
# fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
# while(1):
#     ret, frame = cap.read()
#     fgmask = fgbg.apply(frame)
#     fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
#     cv.imshow('frame',fgmask)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
# cap.release()
# cv.destroyAllWindows()
