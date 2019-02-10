import numpy as np
import cv2


def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel(
            (ksize, ksize), 1.0, theta, 15.0, 0.02, 0, ktype=cv2.CV_32F)
    kern /= 1.5*kern.sum()
    filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


if __name__ == '__main__':
    import sys

    print(__doc__)
    try:
        img_fn = sys.argv[1]
    except:
        img_fn = 'images/example.jpg'

    img = cv2.imread(img_fn)
    if img is None:
        print("Failed to load image file")
        sys.exit(1)

    filters = build_filters()

    res1 = process(img, filters)
    cv2.imshow('result', res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
