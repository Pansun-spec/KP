import cv2
import numpy as np

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    width_new = 1280
    height_new = 720

    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))


    return img_new

img = cv2.imread('900p.jpg')
img_new = img_resize(img)
print(img_new.shape)
print(img.shape)
cv2.imshow('new', img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()