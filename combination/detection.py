import numpy as np
from numpy import array
import cv2 as cv
import imutils
import pickle
import math
import mahotas as mt
import matplotlib.pyplot as plt  # remove later
from typing import List, Set, Dict, Tuple, Optional, Any
np.seterr(divide='ignore', invalid='ignore')

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))

# load the model from disk
filename = '../outputs/finalized_model.sav'
texture_model = pickle.load(open(filename, 'rb'))


# extracts contours of the grey pixels
def get_grey_contours(bgr, type):
    mask_white = np.ones(bgr.shape[:2], dtype="uint8")
    value = bgr.max(axis=2)
    dif = value-bgr.min(axis=2)
    saturation = np.nan_to_num(dif/value)
    mask_white[:, :] = ((value > 200) & (saturation < 0.20))*255
    # cv.imwrite("outputs/" + type + "01_saturationPlusValue.jpg ", mask_white)
    opening = cv.morphologyEx(mask_white, cv.MORPH_OPEN, kernel)
    # cv.imwrite("outputs/" + type + "02_opening.jpg ", opening)  # visualiz
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    # cv.imwrite("outputs/" + type + "03_closing.jpg ", closing)  # visualiz
    result_white = cv.bitwise_and(bgr, bgr, mask=closing)
    # cv.imwrite("outputs/" + type + "04_result_white.jpg", result_white)
    contours = cv.findContours(
        closing.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    return contours


# entering point to color segmentation and contour extraction in some
# cases; returns contours in 2 formats.
def color_segmentation(bgr):
    contours = get_grey_contours(bgr, "full")
    print("Found {} possible smoke clouds in the original image".format(
        len(contours)))
    # for latter search for intersections
    better_format = []
    for c in contours:
        single_pair = []
        for point in c:
            single_pair.append((point[0, 0], point[0, 1]))

        better_format.append(single_pair)

    return contours, better_format


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
            if (positives > len(s_polygon)/10):
                if not any(np.array_equal(
                        first_format[index], arr) for arr in overlapping):
                    overlapping.append(first_format[index])
        index += 1

    return(overlapping)


# Haralick feature extraction
def extract_features(image):
    # calculate haralick texture features
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
    return [leftmost, topmost, rightmost, bottommost]


def resize_contours(input_shape, input_cnt, output_shape):
    output_cnt = []
    for c in input_cnt:
        part = []
        part.append((int((c[0, 0]*output_shape[1])/input_shape[1]), int(
            (c[0, 1] * output_shape[0])/input_shape[0])))
        output_cnt.append(part)
    a = array(output_cnt)
    return a


def detect_smoke(name, frame_stop):
    cap = cv.VideoCapture(name)
    fgbg = cv.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False)
    while(1):
        ret, frame = cap.read()
        # counting what frame it is
        i = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        # stop before the video ends not to give an error
        if frame is None:
            break

        if (i % 10 == 0):
            # applying the bs
            fgmask = fgbg.apply(frame)
            # erosion and dilation
            fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

            # stopping at frame number...
            if i == frame_stop - 10:
                current_frame = frame
                # selecting the color pixels from the foreground
                # cv.imwrite("outputs/00_fullframe.jpg", frame)  # vis
                color = cv.bitwise_and(frame, frame, mask=fgmask)
                # cv.imwrite("outputs/05_color_foreground.jpg", color)  # vis
                contours = get_grey_contours(color, "moved")
                print("Found {} smoke clouds in the foreground".format(
                    len(contours)))

                # now trying to see whether some of the grey regions in the
                # frame are also moving
                original_grey, color_seg_contours = color_segmentation(frame)
                overlapping_contours = check_overlap(
                    color_seg_contours, original_grey, contours)

            # to return the next frame since it might be used in validation
            if i == frame_stop:
                next_frame = frame
    cap.release()
    cv.destroyAllWindows()

    return(next_frame, current_frame, overlapping_contours)


def recognize_smoke(frame, contours):
    index = 0
    recognized_smoke = []
    for c in contours:
        area = cv.contourArea(c)
        print("Area of detected smoke {}: {}".format(index, area))

        # filter out convex regions
        convexity = area/cv.contourArea(cv.convexHull(c))
        print("Convexity of detected smoke {}: {}".format(index, convexity))
        if (not cv.isContourConvex(c) and convexity < 0.95 and area > 100):
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
            prediction = texture_model.predict(
                features.reshape(1, -1))[0]
            print("Predicted class for blob {}: {}".format(index, prediction))

            if (prediction == "maybe-smoke"):
                recognized_smoke.append(c)

                cv.drawContours(frame, [c], -1, (0, 125, 255), 2)
                name = "outputs/06_contour_detection {}.jpg".format(area)
                # cv.imwrite(name, frame)  # visualization
        index += 1

    return recognized_smoke


def validate_smoke(frame, recognized_smoke, validation_picture, photo_taken):
    validated_smoke = []

    # whether a frame or the geotagged image is used
    if (photo_taken):
        image = cv.imread(validation_picture)
    else:
        image = validation_picture
    # cv.imwrite("outputs/07_validation.jpg", image)  # vis

    # resizing contours and/or frame
    # resized_image = cv.resize(frame, (image.shape[1], image.shape[0]))
    res_contours = resize_contours(frame.shape, recognized_smoke, image.shape)

    # detecting the contours in the validation image
    grey_photo, color_seg_contours_photo = color_segmentation(image)
    overlapping_contours_v = check_overlap(
        color_seg_contours_photo, grey_photo, [res_contours])

    index = 0
    for c_v in overlapping_contours_v:
        area_d = cv.contourArea(res_contours)
        area_v = cv.contourArea(c_v)
        print("Areas of validated smoke {}: {}; of detected smoke - {}".format(
            index, area_v, area_d))
        if (area_v > area_d):
            validated_smoke.append(c_v)
            cv.drawContours(image, [c_v], -1, (255, 125, 0), 15)
            cv.drawContours(
                image, [res_contours], -1, (0, 125, 255), 15)
            name = "../outputs/08_cnt_valid_reshaped {}.jpg".format(area_v)
            cv.imwrite(name, image)  # visualization

    return res_contours, validated_smoke


def check_travelled(distances, directions):
    # print(distances)
    margins = []
    results = []
    for d in distances:
        margins.append((d*0.95, d*1.05))

    for i, d in enumerate(distances):
        one_dist_comparison = []
        for j, m in enumerate(margins):
            one_dist_comparison.append(
                (m[0] <= d <= m[1]) and (directions[i] == directions[j]))
        # print(one_dist_comparison)
        results.append(sum(x for x in one_dist_comparison) > 1)

    # results.append(margin[0] <= d <= margin[1] and )
    positives = sum(x for x in results)
    return positives == len(distances)


def direction_loookup(brng):
    directions = [
        "top-right", "right", "bottom-right", "bottom",
        "bottom-left", "left", "top-left", "top"]
    index = brng - 22.5
    if (index < 0):
        index += 360
    index = int(index / 45)
    return(directions[index])


# def d_l(destination_x, origin_x, destination_y, origin_y, max_x, max_y):
def d_l(orig_image, dest_image, max_x, max_y):
    # flipping to cartesian
    dest_cartesian = [dest_image[0] + (max_x/2), (max_y/2) - dest_image[1]]
    orig_cartesian = [orig_image[0] + (max_x/2), (max_y/2) - orig_image[1]]
    deltaX = dest_cartesian[0] - orig_cartesian[0]
    deltaY = dest_cartesian[1] - orig_cartesian[1]
    degrees_final = math.atan2(deltaX, deltaY)/math.pi*180
    if degrees_final < 0:
        degrees_final += 360

    return direction_loookup(degrees_final), round(degrees_final)


def get_direction(first_pos, second_pos, shape):
    directions = []
    distances = []
    print(shape[0])
    i = 0
    for p in first_pos:
        direction = d_l(p, second_pos[i], shape[0], shape[1])

        directions.append(direction)

        # distance
        dist = math.hypot(
            second_pos[i][0] - p[0], second_pos[i][1] - p[1])
        distances.append(round(dist))

        i += 1

    all_travelled = check_travelled(distances, directions)

    return directions, distances, all_travelled
