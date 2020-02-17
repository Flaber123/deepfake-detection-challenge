import cv2


def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    height, width = img.shape[:2]
    if width > height:
        height = height * size // width
        width = size
    else:
        width = width * size // height
        height = size

    resized = cv2.resize(img, (width, height), interpolation=resample)

    return resized


def make_square_image(img):
    height, width = img.shape[:2]
    size = max(height, width)
    top = 0
    bottom = size - height
    left = 0
    right = size - width
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
