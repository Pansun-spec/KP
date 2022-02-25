import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread, imsave
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import Dense, Activation


def flatten_img(img):
    x, y, c = img.shape
    return img.reshape((x * y, c))

img = imread('res.jpg')
plt.figure(figsize=(16, 9))
# plt.imshow(img)
# plt.show()

img_tmp = img.copy()
cv2.rectangle(img_tmp, (4800, 1800), (5400, 2700), (255, 0, 0), -1)
cv2.rectangle(img_tmp, (400, 3300), (2200, 2900), (128, 0, 0), -1)
plt.figure(figsize=(16, 9))
# plt.imshow(img_tmp)
# plt.show()

img_veneer = img[2900:3300, 400:2200, 0:3]
# plt.imshow(img_veneer)
# plt.show()

img_hlv = img[1800:2700, 4800:5400, 0:3]    # highlighted veneer fragment
# plt.imshow(img_hlv)
# plt.show()

img_dsp = imread('dsp.jpg')
# plt.imshow(img_dsp)
# plt.show()

img_dsp_shadow =  img_dsp[:10, :255, :].copy()
for i in range(255):
    img_dsp_shadow[:, i, :] = (img_dsp_shadow[:, i, :] * (i / 256)).astype(np.uint8)
# plt.imshow(img_dsp_shadow)
# plt.show()

img_dsp_shadow = img_dsp_shadow[:, 20:, :]
# plt.imshow(img_dsp_shadow)
# plt.show()

img_dsp_hl = img_dsp[100:110, :255, :].copy()
for i in range(255):
    a = i / 255
    b = 255 * (1 - a)
    img_dsp_hl[:, i, :] = ((img_dsp_hl[:, i, :] * a + b)).astype(np.uint8)

img_dsp_hl = img_dsp_hl[:, 60:, :]
# print(img_dsp_shadow.shape)
# print(img_dsp_hl.shape)
cloud_dsp = np.vstack((flatten_img(img_dsp_shadow), flatten_img(img_dsp_hl)))
# cloud_dsp.shape = (4300, 3) dtype = uint8
# print(cloud_dsp.shape)
# print(cloud_dsp.dtype)
cloud_veneer = np.vstack((flatten_img(img_veneer),flatten_img(img_hlv)))
# cloud_veneer.shape = (1260000, 3) dtype = uint8
# print(img_veneer.shape)
# print(img_hlv.shape)
# print(cloud_veneer.shape)
# print(cloud_veneer.dtype)

cloud_dsp_uniq = np.unique(cloud_dsp, axis=0)
# cloud_dsp_uniq.shape = (3735, 3)
cloud_veneer_uniq = np.unique(cloud_veneer, axis=0)
# cloud_veneer_uniq.shape = (2876, 3)

# fig = plt.figure(figsize=(9, 9))
# ax = Axes3D(fig, azim=70, elev=10)
# ax.scatter(*cloud_dsp_uniq.transpose())
# ax.scatter(*cloud_veneer_uniq.transpose())
# ax.set_xlabel('Red')
# ax.set_xlabel('Green')
# ax.set_xlabel('Blue')
# plt.show()

x_train = np.concatenate((cloud_dsp_uniq, cloud_veneer_uniq))
# x_train.shape = (6611, 3)
# print(x_train.shape)
y_train = np.asarray(
    [[1, 0]] * cloud_dsp_uniq.shape[0] +
    [[0, 1]] * cloud_veneer_uniq.shape[0])
# print(y_train.shape)
# y_train.shape = (6611 ,2)

model = Sequential()
model.add(Dense(2, input_dim=3))
model.add(Activation('softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=20)

img_flatten = flatten_img(img)
pred = model.predict(img_flatten)
pred_binary = np.asarray(list(map(lambda x: 0 if x[0] < x[1] else 1, pred)), dtype=np.uint8)
img_mask = pred_binary.reshape(img.shape[0], img.shape[1])
# plt.figure(figsize=(16, 9))
# plt.imshow(img_mask, cmap='gary')

imsave('mask_perceptron.png', img_mask)


