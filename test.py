import numpy as np
from numpy import array
import cv2 as cv
import imutils
import pickle
import mahotas as mt
import matplotlib.pyplot as plt  # remove later
from georef import getBearing
from typing import List, Set, Dict, Tuple, Optional, Any
np.seterr(divide='ignore', invalid='ignore')

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))

# load the model from disk
filename = 'features/finalized_model.sav'
texture_model = pickle.load(open(filename, 'rb'))

global_smoke = []


# extracts contours of the grey pixels
def get_grey_contours(bgr, type):
    mask_white = np.ones(bgr.shape[:2], dtype="uint8")
    value = bgr.max(axis=2)
    dif = value-bgr.min(axis=2)
    saturation = np.nan_to_num(dif/value)
    mask_white[:, :] = ((value > 200) & (saturation < 0.20))*255
    cv.imwrite(type + "saturationPlusValue.jpg ", mask_white)
    opening = cv.morphologyEx(mask_white, cv.MORPH_OPEN, kernel)
    cv.imwrite(type + "02_opening.jpg ", opening)  # visualization
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    cv.imwrite(type + "03_closing.jpg ", closing)  # visualization
    result_white = cv.bitwise_and(bgr, bgr, mask=closing)
    cv.imwrite(type + "07_result_white.jpg", result_white)  # visualization
    cnts = cv.findContours(
        closing.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts


# entering point to color segmentation and contour extraction in some
# cases; returns contours in 2 formats.
def color_segmentation(bgr):
    cnts = get_grey_contours(bgr, "full")
    print("Found {} possible smoke clouds in the original image".format(
        len(cnts)))
    # for latter search for intersections
    better_format = []
    for c in cnts:
        single_pair = []
        for point in c:
            single_pair.append((point[0, 0], point[0, 1]))

        better_format.append(single_pair)

    return cnts, better_format


# determine if a point is inside a given polygon or not
# polygongon is a list of (x,y) pairs.
def point_inside_polygon(x, y, polygon):
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n+1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def check_overlap(first, first_format, second):
    overlapping = []
    index = 0
    for f_polygon in first:
        for s_polygon in second:
            results = []  # type: List[bool]
            for single in s_polygon:
                inside = point_inside_polygon(
                    single[0, 0], single[0, 1], f_polygon)
                results.append(inside)
            positives = sum(x for x in results)
            if (positives > len(s_polygon)/4):
                if not any(np.array_equal(
                            first_format[index], arr) for arr in overlapping):
                        overlapping.append(first_format[index])
        index += 1

    return(overlapping)


# Haralick (move somewhere else?)
def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean


def get_extremes(cnt):
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    # not the centroid so that there would be less calculations?
    return leftmost, rightmost, topmost, bottommost


def resize_contours(input_shape, input_cnt, output_shape):
    output_cnt = []
    for c in input_cnt:
        part = []
        part.append((int((c[0, 0]*output_shape[1])/input_shape[1]), int(
            (c[0, 1] * output_shape[0])/input_shape[0])))
        output_cnt.append(part)
    a = array(output_cnt)
    return a


# DJI_0899_Trim 200
# YUNC0025_Trim 250, 300 and mod 5
# short, until 110
# DJI_08441, 220
cap = cv.VideoCapture('features/images/short.mp4')
fgbg = cv.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False)
while(1):
    ret, frame = cap.read()
    # counting what frame it is
    i = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    # stop before the video ends not to give an error
    if frame is None:
        break
# (i > 50) &
    if (i % 5 == 0):

        # applying the bs
        fgmask = fgbg.apply(frame)
        # erosion and dilation
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

        # showing (or not showing the result of foreground extraction)
        # cv.imshow('frame', result_white)

        # stopping at frame number...
        if i == 60:
            # selecting the color pixels from the foreground
            cv.imwrite("01_fullframe.jpg", frame)  # visualization
            color = cv.bitwise_and(frame, frame, mask=fgmask)
            cv.imwrite("05_color_foreground.jpg", color)  # visualization
            cnts = get_grey_contours(color, "moved")
            print("Found {} possible smoke clouds in the foreground".format(
                len(cnts)))
            # cv.imshow("Mask", closing)

            # now trying to see whether some of the grey regions in the frame
            # are also moving
            original_grey, color_seg_cnts = color_segmentation(frame)
            overlapping_cnts = check_overlap(
                color_seg_cnts, original_grey, cnts)
            print("Resulting smoke clouds: {}".format(len(overlapping_cnts)))
            # draw the contour and show it
            for c in overlapping_cnts:
                area = cv.contourArea(c)
                print("area of the detected smoke: {}".format(area))
                print(cv.isContourConvex(c))
                # TODO filter out convex
                print(area/cv.contourArea(cv.convexHull(c)))

                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                cimg = np.zeros_like(gray)
                cv.fillPoly(cimg, pts=[c], color=(255, 255, 255))
                # Extract the region of interest
                extracted_pixels = cv.bitwise_and(frame, frame, mask=cimg)
                x, y, w, h = cv.boundingRect(c)
                crop_img = extracted_pixels[y:y+h, x:x+w]
                # cv.imwrite("crop_img.jpg", crop_img) # visualization

                # histogram
                # plt.hist(crop_img.ravel(), 256, [0, 256])
                # plt.show()

                # extract haralick texture from the image
                features = extract_features(crop_img)

                # evaluate the model and predict label
                prediction = texture_model.predict(features.reshape(1, -1))[0]
                print("predicted class: {}".format(prediction))
                # save in global TODO
                if (prediction == "maybe-smoke"):
                    global_smoke.append(c)

                cv.drawContours(frame, [c], -1, (0, 125, 255), 2)
                name = "withcontour_detection {}.jpg".format(area)
                cv.imwrite(name, frame)  # visualization
                cv.imshow("Image", frame)
                cv.waitKey(0)

        # k = cv.waitKey(30) & 0xff
        # if k == 27:
        #         break

    # Validation part TODO
    if (i == 70):
        print(len(global_smoke))
        cv.imwrite("validation.jpg", frame)  # visualization
        # use the frame
        # original_grey_v, color_seg_cnts_v = color_segmentation(frame)
        # overlapping_cnts_v = check_overlap(color_seg_cnts_v, 
        # original_grey_v, global_smoke)

        # or use the geotagged image
        image = cv.imread('inputs/Nebel.jpg')
        # resized_image = cv.resize(frame, (image.shape[1], image.shape[0]))
        old_cnt = resize_contours(frame.shape, global_smoke[0], image.shape)

        original_grey_photo, color_seg_cnts_photo = color_segmentation(image)
        overlapping_cnts_v = check_overlap(
            color_seg_cnts_photo, original_grey_photo, [old_cnt])

        for c_v in overlapping_cnts_v:
            area_v = cv.contourArea(c_v)
            print("area of validation smoke {}".format(area_v))

            leftmost_d, rightmost_d, topmost_d, bottommost_d \
                = get_extremes(old_cnt)
            leftmost_v, rightmost_v, topmost_v, bottommost_v = get_extremes(
                c_v)
            # print("extremes")
            # print(leftmost_d, rightmost_d, topmost_d, bottommost_d)
            # print(leftmost_v, rightmost_v, topmost_v, bottommost_v)

            bearings = getBearing(
                [leftmost_d, rightmost_d, topmost_d, bottommost_d], [
                    leftmost_v, rightmost_v, topmost_v, bottommost_v], "Nebel")
            print(bearings)

            cv.drawContours(image, [c_v], -1, (255, 125, 0), 2)
            cv.drawContours(image, [old_cnt], -1, (0, 125, 255), 2)
            name = "withcontour_validation reshaped{}.jpg".format(area_v)
            cv.imwrite(name, image)  # visualization
            # cv.imshow("Image", image)
            # cv.waitKey(0)

# TODO add a possibility of multiple clouds comparison
cap.release()
cv.destroyAllWindows()
