import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from matplotlib import pyplot as plt
from skimage.io import imread, imsave
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


def cv_show(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def draw_3Dplot(img, mask):
    img_flatten = flatten_img(img)
    mask_flatten = (mask / 255).flatten()

    cloud_dsp = img_flatten[mask_flatten == 0.0]
    cloud_veneer = img_flatten[mask_flatten == 1.0]

    cloud_dsp_uniq = np.unique(cloud_dsp, axis=0)
    cloud_veneer_uniq = np.unique(cloud_veneer, axis=0)

    cloud_veneer_uniq_transpose = cloud_veneer_uniq.transpose()
    cloud_dsp_uniq_transpose = cloud_dsp_uniq.transpose()

    fig = plt.figure(figsize=(9, 9))
    ax = Axes3D(fig, azim=-10, elev=-10)


    ax.scatter(*cloud_dsp_uniq_transpose)
    ax.scatter(*cloud_veneer_uniq_transpose)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    plt.show()


def use_svc(img):
    svc = LinearSVC()
    # Assign parameters from function make_svc to fit model
    svc.coef_ = np.asarray([[0.06359505, 0.02765506, -0.11256832]])
    svc.classes_ = np.asarray([0., 1.])
    svc.intercept_ = np.asarray([0.11000304])

    pred_svc = svc.predict(flatten_img(img))

    img_mask = pred_svc.reshape(img.shape[0], img.shape[1]).astype(np.uint8)

    imsave('mask_svc_f.png', img_mask * 255)


def draw_circle(img):
    mask = imread('mask_svc_f.png')
    # img_tmp = img[:,:1000,:]
    # svc = LinearSVC()
    #
    # for x in range(1_000, 10_000, 1_000):
    #     img_tmp_flatten = flatten_img(img[:, x:x + 1_000, :])
    #     mask_tmp_flatten = (mask[:, x:x + 1_000] / 255).flatten()
    #     svc.fit(img_tmp_flatten, mask_tmp_flatten)
    # pred_svc = svc.predict(flatten_img(img))
    # img_mask = pred_svc.reshape(img.shape[0], img.shape[1]).astype(np.uint8)

    kernel_15 = np.zeros((15, 15), dtype=np.uint8)

    img_mask_erode = cv2.dilate(mask, kernel_15)

    circles = cv2.HoughCircles(
        img_mask_erode,  # 8bit single-channel, grayscale input image
        cv2.HOUGH_GRADIENT,  # Detection method
        2,  # Inverse ratio of the accumulator resolution to the image resolution
        130,  # Minimum distance of the two passed to the Canny edge detector
        param1=100,  # Higher threshold of the two passed to the  Canny edge detector
        param2=20,  # The smaller it is , the more false circles may be detected
        minRadius=5,  # Minimum circle radius
        maxRadius=50) # Maximum circle radius


    img_tmp = img.copy()
    count = 0
    y_pred = []

    for circle in circles[0]:
        center = (int(circle[0]), int(circle[1]))
        cv2.circle(img_tmp, center, 1, (0, 100, 100), 3)
        cv2.circle(img_tmp, center, int(circle[2]), (255, 0, 0), 7)
        cv2.putText(img_tmp, f'{str(count)} radius={round(circle[2] / 5)}mm ', center,
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        print(f"{count + 1}.({circle[0]}, {circle[1]}), diameter: {round(circle[2] / 5)}")

        y_pred.append([circle[0], circle[1], round(circle[2] / 5)])

        count += 1

    img_tmp = ndimage.rotate(img_tmp, -90)
    np.savetxt("pred_circles_f.txt", y_pred, fmt='%d', delimiter=',')
    # plt.figure(figsize=(18, 34))
    # plt.imshow(img_tmp)
    # plt.show()
    imsave('res_detection_f.png', img_tmp)
    cv_show('result', img_tmp)




def accuracy():
    y_true = np.loadtxt('holes_locations.txt', delimiter=',')
    y_pred = np.loadtxt('pred_circles.txt', delimiter=',')

    len_y_pred = len(y_pred)
    len_y_true = len(y_true)
    num_found_holes = len_y_pred / len_y_true

    y_pred_r = np.zeros(99)
    for i in range(0, len_y_pred):
        list_distance = []
        for index in range(0, len_y_true):
            distance = np.sqrt(((y_pred[i][0] - y_true[index][0]) ** 2) + ((y_pred[i][1] - y_true[index][1]) ** 2))
            list_distance.append(distance)
        if min(list_distance) <= 50:  # d<50
            if y_pred_r[list_distance.index(min(list_distance))] == 0:
                y_pred_r[list_distance.index(min(list_distance))] = y_pred[i][2]
            else:
                smaller_r = min(y_pred_r[list_distance.index(min(list_distance))], y_pred[i][2])
                y_pred_r[list_distance.index(min(list_distance))] = smaller_r

    y_true_r = []
    for i in range(0, len_y_true):
        y_true_r.append(y_true[i][2])
    y_true_r = np.array(y_true_r)

    print(f"accuracy of finding the num of holes: {num_found_holes}\n")
    return y_true_r, y_pred_r

if __name__ == "__main__":
    img = imread('res.jpg')
    gen_train_labels(img)
    mask = imread('mask_f.png')
    # Create a mask

    # draw_3Dplot(img, mask)
    # Axes3D

    use_svc(img)

    draw_circle(img)

    y_true_r, y_pred_r = accuracy()

    print(f"mean_squared_error: {mean_squared_error(y_true_r, y_pred_r)}\n")
    print(f"mean_absolute_percentage_error: {mean_absolute_percentage_error(y_true_r, y_pred_r)}")
