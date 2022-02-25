import cv2
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from sklearn.svm import LinearSVC, SVC

def flatten_img(img):
    x, y, c = img.shape
    return img.reshape((x*y, c))


def gen_train_labels(img):
    df = pd.read_csv('holes_location.csv')
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # Create a black image the same size as img

    for index, row in df.iterrows():
        x = (int)(10 * row['x'])
        y = (int)(10 * row['y'])
        r = (int)( 5 * row['d'])
        cv2.circle(mask, (x, y), r, (255,), -1)

    imsave('mask_f.png', mask)


if __name__ == "__main__":
    img = imread('res.jpg')
    gen_train_labels(img)

    mask = imread('mask_f.png')
    svc = LinearSVC()
    for x in range(1_000, 10_000, 1_000):
        img_tmp_flatten = flatten_img(img[:, x:x + 1_000, :])
        mask_tmp_flatten = (mask[:, x:x + 1_000] / 255).flatten()
        svc.fit(img_tmp_flatten, mask_tmp_flatten)

    print(f"{svc.coef_}\n")
    print(f"{svc.classes_}\n")
    print(f"{svc.intercept_}\n")