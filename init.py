from detection import detect_smoke, recognize_smoke, validate_smoke
from detection import get_extremes, get_direction
from georef import get_bearing, start_georeferencing

# DJI_0899_Trim 200
# YUNC0025_Trim 250, 300 and mod 5
# short, until 110
# DJI_08441, 220
next_frame, frame, detected_contours = detect_smoke('inputs/short.mp4', 70)
print("Detected smoke clouds: {}".format(len(detected_contours)))

recognized_contours = recognize_smoke(frame, detected_contours)
print("Recognized smoke clouds: {}".format(len(recognized_contours)))

for r_c in recognized_contours:
    # res_recognized, validated_contours = validate_smoke(
        # frame, r_c, 'inputs/Nebel.jpg', True)
    res_recognized, validated_contours = validate_smoke(
        frame, r_c, next_frame, False)
    if (validated_contours):
        if (len(validated_contours) == 1):
            # getting the 4 corners
            extremes_d = get_extremes(res_recognized)
            extremes_v = get_extremes(validated_contours[0])
            print(extremes_d)
            print(extremes_v)

            # start_georeferencing("Nebel")
            # bearings = get_bearing(extremes_d, extremes_v, "Nebel")
            # print(bearings)
            directions, distances, all_travelled = get_direction(
                extremes_d, extremes_v)
            print(directions)
            print(distances)
            print(all_travelled)

        else:
            print(len(validated_contours))
