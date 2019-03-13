
import subprocess
from subprocess import Popen, PIPE
import sys
import os
import math
import re
# exiftool.exe and cs2cs.exe must be installed
# set PROJ_LIB= {{folder with epsg file}} http://svn.osgeo.org/metacrs/proj/trunk/proj/nad/epsg
# pyproj is faster than cs2cs?

sensor_height = 0.88
focal_length = 0.88
image_height = 3648
image_width = 5472
sensor_width = 1.32
pi = 3.14159265359

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def dms2dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    if direction == 'W' or direction == 'S':
        dd *= -1
    return dd

def parse_dms(dms):
    parts = re.split('[^\d\w\.]+', dms)
    deg = dms2dd(parts[0], parts[2], parts[3], parts[4])
    return round(deg, 5)


# reading out the exif data
def getExif(filename): 
    exifdata = subprocess.check_output(["exiftool.exe", filename], shell=True)
    # saving to a dictionary
    exifdata = exifdata.splitlines()
    exif = dict()
    for i, each in enumerate(exifdata):
        # tags and values are separated by a colon
        tag,val = each.decode().split(': ', 1) # '1' only allows one split
        exif[tag.strip()] = val.strip()
    lat = parse_dms(exif['GPS Latitude'])
    lon = parse_dms(exif['GPS Longitude'])
    alt = num(exif['Relative Altitude'])
    yaw = num(exif['Gimbal Yaw Degree'])
    return(lat, lon, alt, yaw)

# LB to UTM
def lbToUTM(lat, lon, alt):
    input_string = '     {}      {}      {} '.format(lon,lat,alt)
    text_file = open("LB.txt", "w")
    text_file.write(input_string)
    text_file.close()
    os.environ["PROJ_LIB"] = "C:/PROJSHARE" #!!!!!
    # p = subprocess.Popen(['cs2cs.exe', '+init=epsg:4326', '+to', '+init=epsg:25832', 'LB.txt', 'PC.txt'])
    # print(p)
    os.system('"cs2cs.exe +init=epsg:4326 +to +init=epsg:25832 < LB.txt > PC.txt"')
    with open('PC.txt', encoding='utf8') as f:
        output = f.read().strip().split()
        return num(output[0]), num(output[1])

# # write the worldfile maybe
# # worldfile order
# a = 0.00241
# d = -137.60 # the other way?
# b = 137.60
# e = -0.00241
# c = 405598.673
# f = 5724801.333

# to calculate size of a pixel
def getPixelSize(flight_height):
    flight_height_cm = flight_height*100
    GSDh = (flight_height_cm*sensor_height)/(focal_length*image_height)
    GSDw = (flight_height_cm*sensor_width)/(focal_length*image_width)
    print(GSDh)
    print(GSDw)
    return round(GSDw, 3) if GSDw > GSDh else round(GSDh, 3)

# to calculate coordinates of a pixel
def pixel2coord(a, d, b, e, c, f, col, row):
    xp = a * col + b * row + c
    yp = d * col + e * row + f
    return(xp, yp)

def createWorldFile(pixel_size, rotation, lat_mid, lon_mid):
    a = pixel_size * math.cos((pi/180)*rotation)
    d = -pixel_size * math.sin((pi/180)*rotation)
    b = -pixel_size * math.sin((pi/180)*rotation)
    e = -pixel_size * math.cos((pi/180)*rotation)
    # coord of the (0,0) pixel
    c, f = pixel2coord(a, d, b, e, lat_mid, lon_mid, -image_width/2, -image_height/2)
    print(a, d, b, e, c, f)

filename = "georef/DJI_0836.JPG"
lat_LM, lon_LM, altitude, rotation = getExif(filename)
print(lat_LM, lon_LM, altitude, rotation)
lon_UTM, lat_UTM = lbToUTM(lat_LM, lon_LM, altitude)
pixel_size = getPixelSize(altitude)/100
createWorldFile(pixel_size, rotation, lon_UTM, lat_UTM)

