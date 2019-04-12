import cv2
from colour import simple_gray, gray_plus_intensity
from colour import saturation_plus_value, inYCbCrColourSpace
from haralick import train_feature_model
from motion import backgounrd_subtraction, optical_flow, corner_tracking
from wavelets import w1d, w2d, details
from etc import detect_edges, fourier_transform, histogram_contours, grabcut

# Read out the image
path = "../inputs/DJI_0843_frame00064.JPG"
image = cv2.imread('../inputs/DJI_0843_frame00064.JPG')
cap = cv2.VideoCapture('../inputs/DJI_0843_small.mp4')


# # Colour segmentation
# # In HSV
# simple_gray(image)
# # In RGB
# gray_plus_intensity(image)
# # Only SV
# saturation_plus_value(image)
# # In YCbCr
# inYCbCrColourSpace(image)

# # Motion detection
# backgounrd_subtraction(cap)
# optical_flow(cap)
# corner_tracking(cap)

# #wavelets
# w2d(path, 'db1', 9)
# details(path)
# w1d('../inputs/DJI_0843_small.mp4')

# # edge detection
# detect_edges(image)
# fourier_transform(path)
# grabcut(image)
# histogram_contours(image)

# # Texture model training
# train_feature_model()
