import cv2
import numpy
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def data_get(path):
    f = open(path, mode='r')
    dets = f.readline().strip().split(' ')
    y_true = []
    for i in dets:
        y_true.append(int(i))
    return y_true


def image_prc(image):
    img_gau_blur = cv2.GaussianBlur(image, (9, 9), 2)  # Gaussian Сгладить
    cv_show('img_gau_blur', img_gau_blur)
    img_gray = cv2.cvtColor(img_gau_blur, cv2.COLOR_BGR2GRAY)  # Преобразовать изображение в полутоновое
    cv_show('img_gary', img_gray)
    img_blur = cv2.blur(img_gau_blur, (4, 4))  # Сгладить

    img_canny = cv2.Canny(img_blur, 50, 150)    # Применить детектор границ Canny
    cv_show('canny', img_canny)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(img_canny, cv2.MORPH_GRADIENT, kernel)  # Morphological Gradient
    # cv_show('closing', closing)

    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    img_contours = cv2.drawContours(img_canny, contours, -1, (255, 255, 255), 2)  # draw contours
    # cv_show('conts', img_contours)

    length = len(contours)
    # Залить замкнутые области
    for i in range(0, length):
        cv2.fillPoly(img_contours, [contours[i]], (255, 255, 255))
    cv_show('res', img_contours)
    return img_contours


def hough_search(image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 200, 255, 255, 18, 0)
    # dp, minDist, param1_Canny threshold, param2 center threshold, minr, maxr
    circles = np.uint16(np.around(circles))

    count = 0
    r_pred = []
    y_pred = []
    for i in circles[0, :]:
        # if i[2] > 100 or i[2] < 10:
        #     continue
        # else:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        y_pred.append([i[0], i[1], i[2]])
        r_pred.append(round(i[2] / 10) * 10)
        r_pred.sort()
        print(f"{count + 1}.({i[0]}, {i[1]}), d-detected: {i[2]}(pixel)")
        count += 1
        cv2.putText(img, f'{str(count)} radius={int(round(i[2] / 10) / 2 * 10)}mm ', (i[0], i[1]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)

    np.savetxt("pred_circles.txt", y_pred, fmt='%d', delimiter=',')

    cv_show('final result', img)
    return r_pred


def list_proc(list_pred, list_true):
    n = len(list_true)
    pred_full = np.zeros(n, dtype=np.int16)
    for i in list_pred:
        for idx, val in enumerate(list_true):
            if i == val or abs(i - val) < 5:
                pred_full[idx] = i
    return pred_full


def found_hole_percent(len_pred, len_true):
    if len_pred < len_true:
        found_hole_percent = len_pred / len_true
    else:
        found_hole_percent = len_true / len_pred
    print(f"found_hole_percent: {found_hole_percent}")  # how many circles have been found?

def data_proc(y_pred, y_true):
    y_pred_r = [0, 0, 0, 0, 0, 0]
    for i in range(0, len(y_pred)):
        list_distance = []
        for index in range(0, len(y_true)):
            distance = numpy.sqrt(
                ((y_pred[i][0] - y_true[index][0]) ** 2) + ((y_pred[i][1] - y_true[index][1]) ** 2))
            list_distance.append(distance)
        if min(list_distance) <= 10:  # d<10
            if y_pred_r[list_distance.index(min(list_distance))] == 0:
                y_pred_r[list_distance.index(min(list_distance))] = y_pred[i][2]
            else:
                smaller_r = min(y_pred_r[list_distance.index(min(list_distance))], y_pred[i][2])
                y_pred_r[list_distance.index(min(list_distance))] = smaller_r
    count = 0
    for m in range(0, len(y_pred_r)):
        if y_pred_r[m] == 0:
            count = count + 1
    for n in range(0, count):
        y_pred_r.remove(0)
    print(f"y_pred_r: {y_pred_r}")

    # find the y_true diameter
    y_true_r = []
    for i in range(0, len(y_true)):
        y_true_r.append(y_true[i][2])
    print(f"y_true_r: {y_true_r}")
    return y_true_r, y_pred_r

if __name__ == "__main__":
    img = cv2.imread('for_search.jpg')

    img_contours = image_prc(img)
    pred_lose = hough_search(img_contours)
    # print(f"pred_lose: {pred_lose}")
    path_true = "/home/kehua/PycharmProjects/KP/project/circles_generate/circles_pos.txt"  # path of circle_pos.txt
    y_true = np.loadtxt(path_true, delimiter=',')
    print(f"y_true:\n {y_true}")

    path_pred = "/home/kehua/PycharmProjects/KP/project/circle_search/pred_circles.txt"
    y_pred = np.loadtxt(path_pred, delimiter=',')
    print(f"y_pred:\n {y_pred}")

    len_pred = len(y_pred)
    len_true = len(y_true)
    found_hole_percent(len_pred, len_true)

    y_true_r, y_pred_r = data_proc(y_pred, y_true)

    # accuracy
    mean_squared_error = mean_squared_error(y_true_r, y_pred_r)
    print(f"mean_squared_error: {mean_squared_error}")
    mean_absolute_percentage_error = mean_absolute_percentage_error(y_true_r, y_pred_r)
    print(f"mean_absolute_percentage_error: {mean_absolute_percentage_error}")
