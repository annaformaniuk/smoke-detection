
import subprocess
from subprocess import Popen, PIPE
import sys
import os
# exiftool.exe and cs2cs.exe must be installed
# set PROJ_LIB= {{folder with epsg file}} http://svn.osgeo.org/metacrs/proj/trunk/proj/nad/epsg
# pyproj is faster than cs2cs?

# # reading out the exif data
# filename = "DJI_0836.JPG"
# exifdata = subprocess.check_output(['exiftool.exe', filename], shell=True)
# # saving to a dictionary
# exifdata = exifdata.splitlines()
# exif = dict()
# for i, each in enumerate(exifdata):
#      # tags and values are separated by a colon
#      tag,val = each.decode().split(': ', 1) # '1' only allows one split
#      exif[tag.strip()] = val.strip()
# print(exif['Compression'])

# LB to UTM
os.environ["PROJ_LIB"] = "C:/PROJSHARE" #!!!!!
# p = subprocess.Popen(['cs2cs.exe', '+init=epsg:4326', '+to', '+init=epsg:25832', 'LB.txt', 'PC.txt'])
# print(p)
os.system('"cs2cs.exe +init=epsg:4326 +to +init=epsg:25832 < LB.txt > PC.txt"')

# # write the worldfile maybe
# # worldfile order
# a = 0.00241
# d = -137.60 # the other way?
# b = 137.60
# e = -0.00241
# c = 405598.673 # change for real ones
# f = 5724801.333

c = 1145229.31
a = 2.41
b = 137.60
f = 5693390.12
d = -137.60
e = -2.41

# to calculate coordinates of a pixel
def pixel2coord(col, row):
    xp = a * col + b * row + c
    yp = d * col + e * row + f
    return(xp, yp)


# x,y = pixel2coord(-2736,-1824)
# print(x, y)