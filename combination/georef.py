
import subprocess
from subprocess import Popen, PIPE
import sys
import os
import math
import re
import decimal
import json

# exiftool.exe and cs2cs.exe must be installed
# set PROJ_LIB= {{folder with epsg file}}
# http://svn.osgeo.org/metacrs/proj/trunk/proj/nad/epsg

# create a new context for this task
ctx = decimal.Context()
ctx.prec = 20

sensor_height = 0.88
focal_length = 0.88
image_height = 3648
image_width = 5472
sensor_width = 1.32


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def to_string(f):
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')


def dms_to_dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    if direction == 'W' or direction == 'S':
        dd *= -1
    return dd


def parse_dms(dms, replace: False, rad: False):
    if (replace):
        dms = dms.replace("d", " deg ")
    parts = re.split('[^\d\w\.]+', dms)
    deg = dms_to_dd(parts[0], parts[2], parts[3], parts[4])
    if (rad):
        return math.radians(deg)
    else:
        return round(deg, 5)


# reading out the exif data
def get_exif(filename):
    exifdata = subprocess.check_output(["exiftool.exe", filename], shell=True)
    # saving to a dictionary
    exifdata = exifdata.splitlines()
    exif = dict()
    for i, each in enumerate(exifdata):
        # tags and values are separated by a colon
        tag, val = each.decode().split(': ', 1)  # '1' only allows one split
        exif[tag.strip()] = val.strip()
    lat = parse_dms(exif['GPS Latitude'], False, False)
    lon = parse_dms(exif['GPS Longitude'], False, False)
    alt = num(exif['Relative Altitude'])
    yaw = num(exif['Gimbal Yaw Degree'])
    return(lat, lon, alt, yaw)


# LB to UTM
def lb_to_UTM(lat, lon, alt):
    input_string = '     {}      {}      {} '.format(lon, lat, alt)
    text_file = open("LB.txt", "w")
    text_file.write(input_string)
    text_file.close()
    os.environ["PROJ_LIB"] = "C:/PROJSHARE"  # !!!!!
    # p = subprocess.Popen(['cs2cs.exe', '+init=epsg:4326', '+to',
    # '+init=epsg:25832', 'LB.txt', 'PC.txt'])
    # print(p)
    os.system(
        '"cs2cs.exe +init=epsg:4326 +to +init=epsg:25832 < LB.txt > PC.txt"')
    with open('PC.txt', encoding='utf8') as f:
        output = f.read().strip().split()
        return num(output[0]), num(output[1])


def UTM_to_lb(lat, lon, alt):
    input_string = '     {}      {}      {} '.format(lat, lon, alt)
    text_file = open("LB2.txt", "w")
    text_file.write(input_string)
    text_file.close()
    os.environ["PROJ_LIB"] = "C:/PROJSHARE"  # !!!!!
    os.system(
        '"cs2cs.exe +init=epsg:25832 +to +init=epsg:4326 < LB2.txt > PC2.txt"')
    with open('PC2.txt', encoding='utf8') as f:
        output = f.read().strip().split()
        return output[0], output[1]


# to calculate size of a pixel
def get_pixel_size(flight_height):
    flight_height_cm = flight_height*100
    GSDh = (flight_height_cm*sensor_height)/(focal_length*image_height)
    GSDw = (flight_height_cm*sensor_width)/(focal_length*image_width)
    return round(GSDw, 3) if GSDw > GSDh else round(GSDh, 3)


# to calculate coordinates of a pixel
def pixel_to_coord(a, d, b, e, c, f, col, row):
    xp = a * col + b * row + c
    yp = d * col + e * row + f
    return(xp, yp)


def create_worldFile(pixel_size, rotation, lat_mid, lon_mid, name):
    a = pixel_size * math.cos(math.radians(rotation))
    d = -pixel_size * math.sin(math.radians(rotation))
    b = -pixel_size * math.sin(math.radians(rotation))
    e = -pixel_size * math.cos(math.radians(rotation))
    # coord of the (0,0) pixel
    c, f = pixel_to_coord(
        a, d, b, e, lat_mid, lon_mid, -image_width/2, -image_height/2)
    print("worldfile info:")
    print(a, d, b, e, c, f)
    worldfile = [to_string(a), to_string(
        d), to_string(b), to_string(e), to_string(c), to_string(f)]
    filename = "../inputs/{}.jgw".format(name)
    text_file = open(filename, "w")
    for line in worldfile:
        text_file.write(line)
        text_file.write("\n")
    text_file.close()


def direction_loookup(brng):
    bearings = ["NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    index = brng - 22.5
    if (index < 0):
        index += 360
    index = int(index / 45)
    return(bearings[index])


# http://www.movable-type.co.uk/scripts/latlong.html
def get_bearing(first_pos, second_pos, file_name):
    exists = os.path.isfile('../inputs/{}.jgw'.format(file_name))
    if exists:
        print("WorldFile exists - {}".format(exists))
        values = []
        value_strings = [line.rstrip('\n') for line in open(
            '../inputs/{}.jgw'.format(file_name))]
        for v in value_strings:
            values.append(num(v))
        first_coord = []
        second_coord = []
        first_deg = []
        second_deg = []
        for pos in first_pos:
            coordx, coordy = pixel_to_coord(
                values[0], values[1], values[2], values[3], values[4], values[
                    5], pos[0], pos[1])
            coordx_lb, coordy_lb = UTM_to_lb(coordx, coordy, 10)
            first_deg.append((parse_dms(coordx_lb, True, False), parse_dms(
                coordy_lb, True, False)))
            first_coord.append((parse_dms(coordx_lb, True, True), parse_dms(
                coordy_lb, True, True)))
        for pos in second_pos:
            coordx, coordy = pixel_to_coord(values[0], values[1], values[
                2], values[3], values[4], values[5], pos[0], pos[1])
            coordx_lb, coordy_lb = UTM_to_lb(coordx, coordy, 10)
            second_deg.append((parse_dms(coordx_lb, True, False), parse_dms(
                coordy_lb, True, False)))
            second_coord.append((parse_dms(coordx_lb, True, True), parse_dms(
                coordy_lb, True, True)))

        directions = []
        for i, coord in enumerate(first_coord):
            y = math.sin(second_coord[i][1]-coord[1]) * math.cos(
                second_coord[i][0])
            x = math.cos(coord[0])*math.sin(second_coord[i][0]) - math.sin(
                coord[0])*math.cos(second_coord[i][0])*math.cos(
                    second_coord[i][1]-coord[1])
            brng = math.degrees(math.atan2(y, x))
            directions.append((direction_loookup(brng), round(brng)))
        # print(directions)
        return first_deg, second_deg, directions
    else:
        print("nofile")
        start_georeferencing(file_name)
        get_bearing(first_pos, second_pos, file_name)


def start_georeferencing(name):
    filename = "../inputs/" + name + ".JPG"
    lat_LM, lon_LM, altitude, rotation = get_exif(filename)
    print("lat, lon, alt, rotation:")
    print(lat_LM, lon_LM, altitude, rotation)
    lon_UTM, lat_UTM = lb_to_UTM(lat_LM, lon_LM, altitude)
    pixel_size = get_pixel_size(altitude)/100
    create_worldFile(pixel_size, rotation, lon_UTM, lat_UTM, name)


def saveJson(first, second):
    first.append(first[0])
    second.append(second[0])
    with open('test.json', 'w') as file:
        json.dump({'type': "FeatureCollection", "features": [{
            "type": "Feature", "geometry": {
                "type": "Polygon", "coordinates": [first]}, "style": {
                    "fill": "red"}, "properties": {"name": "second"}}, {
            "type": "Feature", "geometry": {
                "type": "Polygon", "coordinates": [second]}, "style": {
                    "fill": "blue"}, "properties": {"name": "second"}}]}, file)
