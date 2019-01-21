import matplotlib.pyplot as plt
import numpy as np
import cv2

# from Yu C., A real-time video fire flame and smoke detection algorithm
def test(rgb, gray):
    image_new = np.ones(gray.shape[:2], dtype="uint8")
    m = np.ones(gray.shape[:2], dtype="uint8")
    n = np.ones(gray.shape[:2], dtype="uint8")
    i = np.ones(gray.shape[:2], dtype="uint8")

    m = rgb.max(axis=2)
    n = rgb.min(axis=2)
    i[:, :] = (rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2])/3
    cv2.imwrite("i.jpg", i)
    print(np.amax(i))
    print(i)

    # counter2 = np.sum(np.bitwise_and(img < 10, img > 0))
    # counter = np.sum(m-n < 20)  # Sums work on binary values

    # This is 0 or 1 depending on whether it is == 0
    image_new[:, :] = (i > 80) & (m - n < 20)

    # So scale the values up with a simple multiplcation
    image_new = image_new*255  # image_new[i,j] = image_new[i,j]*255
    cv2.imwrite("end.jpg", image_new)


image = cv2.imread('DJI_0843_frame00064.jpg')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # with grayscale
# hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS) # or intensity?
# lightness = hls_image[:,:,1]
test = test(rgb_image, gray)

# #to simply segment all the grey pixels
# light_white = (0, 0, 200)
# dark_white = (145, 60, 255)
# mask_white = cv2.inRange(hsv_image, light_white, dark_white)
# result_white = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_white)

# plt.subplot(1, 2, 1)
# plt.imshow(mask_white, cmap="gray")
# plt.subplot(1, 2, 2)

# blur = cv2.GaussianBlur(result_white, (7, 7), 0)

# plt.imshow(blur)
# plt.show()
