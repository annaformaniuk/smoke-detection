import cv2
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


# to simply segment all the gray pixels
def simpleGray(bgr):
    hsv_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    light_white = (0, 0, 200)
    dark_white = (145, 60, 255)
    mask_white = cv2.inRange(hsv_image, light_white, dark_white)
    result_white = cv2.bitwise_and(bgr, bgr, mask=mask_white)
    cv2.imwrite("hsvGray.jpg", result_white)

    plt.subplot(1, 2, 1)
    plt.imshow(mask_white, cmap="gray")
    plt.subplot(1, 2, 2)

    blur = cv2.GaussianBlur(result_white, (7, 7), 0)
    plt.imshow(blur)
    plt.show()

    # # Calculate histogram with mask and without mask
    # # Check third argument for mask
    # hist_full = cv2.calcHist([bgr], [0], None, [256], [0, 256])
    # hist_mask = cv2.calcHist([bgr], [0], mask_white, [256], [0, 256])

    # plt.subplot(221), plt.imshow(bgr, 'gray')
    # plt.subplot(222), plt.imshow(mask_white, 'gray')
    # plt.subplot(223), plt.imshow(result_white, 'gray')
    # plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    # plt.xlim([0, 256])
    # plt.show()


# from Yu C., A real-time video fire flame and smoke detection algorithm
def grayPlusIntensity(bgr, gray, i):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image_new = np.ones(gray.shape[:2], dtype="uint8")
    m = np.ones(gray.shape[:2], dtype="uint8")
    n = np.ones(gray.shape[:2], dtype="uint8")
    # i = np.ones(gray.shape[:2], dtype="uint8")

    m = rgb.max(axis=2)
    n = rgb.min(axis=2)
    # i[:, :] = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2])/3
    # cv2.imwrite("i.jpg", i)
    print(np.amax(i))
    print(i)

    # counter2 = np.sum(np.bitwise_and(img < 10, img > 0))
    # counter = np.sum(m-n < 20)  # Sums work on binary values

    # This is 0 or 1 depending on whether it is == 0
    image_new[:, :] = (i > 200) & (m - n < 20)

    # So scale the values up with a simple multiplcation
    image_new = image_new*255  # image_new[i,j] = image_new[i,j]*255
    cv2.imwrite("grayPlusIntensity.jpg", image_new)
    result_white = cv2.bitwise_and(bgr, bgr, mask=image_new)

    plt.subplot(1, 2, 1)
    plt.imshow(result_white)
    plt.subplot(1, 2, 2)

    blur = cv2.GaussianBlur(image_new, (7, 7), 0)
    plt.imshow(blur)
    plt.show()


# from Shafar, Early Smoke Detection on Video Using Wavelet Energy
def saturationPlusValue(rgb):
    image_new = np.ones(gray.shape[:2], dtype="uint8")
    value = np.ones(gray.shape[:2], dtype="uint8")
    value = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    dif = value-minimum
    saturation = np.ones(gray.shape[:2], dtype="uint8")

    saturation = np.nan_to_num(dif/value)
    print(value)

    image_new[:, :] = (value > 230) & (saturation < 0.20)
    image_new = image_new*255
    cv2.imwrite("saturationPlusValue.jpg", image_new)

# from Wang, Video smoke detection using shape, color, and dynamic features
def inYCbCrColourSpace(bgr):
    image_new = np.ones(bgr.shape[:2], dtype="uint8")
    image_ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCR_CB)

    image_new[:, :] = (image_ycrcb[:, :, 1] > 115) & (image_ycrcb[:, :, 1] < 141) & (
        image_ycrcb[:, :, 2] > 115) & (image_ycrcb[:, :, 2] < 141) & (image_ycrcb[:, :, 0] > 220)
    image_new = image_new*255
    cv2.imwrite("inYCbCrColourSpace.jpg", image_new)


image = cv2.imread('images/YUNC0025.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # with grayscale
# hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS) # or intensity?
# lightness = hls_image[:,:,1]
# grayPlusIntensity(image, gray, lightness)
# saturationPlusValue(rgb_image)
# inYCbCrColourSpace(image)
simpleGray(image)
