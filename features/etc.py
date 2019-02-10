import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

img = cv2.imread('images/DJI_0843_frame00064.jpg')

# # detecting edges
# edges = cv2.Canny(img,100,200)

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# plt.show()

# # Fourier Transform
# dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)

# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

# grabcut 
# img = cv2.imread('DJI_0806_frame00132.jpg')
# mask = np.zeros(img.shape[:2],np.uint8)
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
# rect = (50,50,450,290)
# cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask2[:,:,np.newaxis]
# plt.imshow(img),plt.colorbar(),plt.show()

# # hsv histogram
# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# plt.imshow(hist,interpolation = 'nearest')
# plt.show()

# shape
lower = np.array([230, 230, 230])
upper = np.array([255, 255, 255])
shapeMask = cv2.inRange(img, lower, upper)

# find the contours in the mask
cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("I found {} white shapes".format(len(cnts)))
cv2.imshow("Mask", shapeMask)
 
# loop over the contours
for c in cnts:
	# draw the contour and show it
	cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
	cv2.imshow("Image", img)
	cv2.waitKey(0)