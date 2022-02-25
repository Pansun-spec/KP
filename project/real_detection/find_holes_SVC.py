import cv2
import numpy as np
from skimage.io import imread, imsave
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC, SVC
from scipy import ndimage

def flatten_img(img):
    x, y, c = img.shape
    return img.reshape((x * y, c))

def find_holes_svc(img, mask):
    img_tmp = img[:, :1000, :]
    svc = LinearSVC()
    for x in range(1_000, 10_000, 1_000):
        img_tmp_flatten = flatten_img(img[:, x:x + 1_000, :])
        mask_tmp_flatten = (mask[:, x:x + 1_000] / 255).flatten()
        svc.fit(img_tmp_flatten, mask_tmp_flatten)

    pred_svc = svc.predict(flatten_img(img))
    img_mask = pred_svc.reshape(img.shape[0], img.shape[1]).astype(np.uint8)

    kernel_15 = np.zeros((15, 15), dtype=np.uint8)

    img_mask_erode = cv2.dilate(img_mask, kernel_15)