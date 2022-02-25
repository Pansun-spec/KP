import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.io import imread

img = imread('res.jpg')
# plt.imshow(img)
# plt.show()

df = pd.read_csv('holes_location.csv')
print(df)

img_tmp = img.copy()

for index, row in df.iterrows():
    x = (int)(10 * row['x'])
    y = (int)(10 * row['y'])
    r = (int)(5 * row['d'])
    n = str(int(row['n']))
    cv2.circle(img_tmp, (x, y), r, (255, 0, 0), 7)
    cv2.putText(img_tmp, n, (x + r + 2, y), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 0, 0))

img_tmp = ndimage.rotate(img_tmp, -90)  # slow rotation based on affine transform

plt.figure(figsize=(18, 34))
plt.imshow(img_tmp)
plt.show()

cv2.imwrite('res_dectected_byhuman.jpg', img_tmp)

