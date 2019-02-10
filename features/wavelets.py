import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple, Optional
import scipy.interpolate as interp

# get back to https://stackoverflow.com/questions/24536552/how-to-combine-pywavelet-and-opencv-for-image-processing

big_array = []  # type: List[int]
actual_data = [] # type: List[int]
time_array = [] # type: List[int]

def w2d(img, mode='haar', level=1):
    imArray = cv2.imread(img)
    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    # normalize
    imArray /= 255

    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)
    #Display result
    cv2.imshow('image',imArray_H)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def three(img):
    imArray = cv2.imread(img)
    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    # normalize
    imArray /= 255

    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(imArray, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

def w1d(img):
    cap = cv2.VideoCapture(img)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            global big_array, time_array, actual_data

            window = np.matrix(gray[300:325,300:325])
    # # VISUALIZATION
    #         cv2.imshow('frame', window)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    # cap.release()
    # cv2.destroyAllWindows()

            # actual_data.append(gray[600,600])
            big_array.append(window.mean())
            time_array.append(cap.get(0))
        else:
            break

    print(len(big_array))
    print(len(time_array))
    # (cA, cD) = pywt.dwt(big_array, 'db1') #Single level Discrete Wavelet Transform.
    coeffs = pywt.wavedec(big_array, 'db1', level=2)
    cA2, cD2, cD1 = coeffs
    plt.plot(big_array)
    #  to plot the cA and cD coefficients in time, just reduce time resolution by 2
    asArray = np.array(big_array)
    cD1_interpr = interp.interp1d(np.arange(cD1.size),cD1)
    cD1_stretch = cD1_interpr(np.linspace(0,cD1.size-1,asArray.size))
    plt.plot(cD1_stretch)
    plt.show()

w2d("images/DSC00654.jpg",'db1',9)
# three("images/YUNC0025.jpg")
# print(pywt.wavelist())
# w1d('images/DJI_0843.mp4')
