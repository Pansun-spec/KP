import cv2
import numpy as np

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    width_new = 1280

    img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    return img_new

img = cv2.imread('540p.jpg')
img_new = img_resize(img)
print(img.shape)
print(img_new.shape)
cv2.imshow('new', img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()