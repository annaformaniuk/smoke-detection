from detection import detect_smoke, recognize_smoke, validate_smoke
from detection import get_extremes, get_direction
from georef import get_bearing, start_georeferencing, saveJson
import cv2 as cv

# DJI_0899_Trim 200
# YUNC0025_Trim 250, 300 and mod 5
# short, until 110
# DJI_08441, 220
next_frame, frame, detected_contours = detect_smoke('inputs/short.mp4', 70)
print("Detected smoke clouds: {}".format(len(detected_contours)))
cv.imwrite("outputs/first.jpg", frame)  # vis
cv.imwrite("outputs/next_frame.jpg", next_frame)  # vis

recognized_contours = recognize_smoke(frame, detected_contours)
print("Recognized smoke clouds: {}".format(len(recognized_contours)))

for r_c in recognized_contours:
    res_recognized, validated_contours = validate_smoke(
        frame, r_c, 'inputs/Nebel.jpg', True)
    # res_recognized, validated_contours = validate_smoke(
    # frame, r_c, next_frame, False)
    if (validated_contours):
        if (len(validated_contours) == 1):
            # getting the 4 corners
            extremes_d = get_extremes(res_recognized)
            extremes_v = get_extremes(validated_contours[0])
            print("Detected corners: {}".format(extremes_d))
            print("Validated corners: {}".format(extremes_v))

            start_georeferencing("Nebel")
            first, second, bearings = get_bearing(
                extremes_d, extremes_v, "Nebel")
            print("Geographical bearings: {}".format(bearings))
            saveJson(first, second)

            directions, distances, all_travelled = get_direction(
                extremes_d, extremes_v)
            print("Direction of travel on image: {}".format(directions))
            print("Travelled distances: {}".format(distances))
            print("Have all points travelled in one direction? - {}".format(
                all_travelled))

        else:
            print(len(validated_contours))
