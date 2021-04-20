import cv2
import numpy as np
import math
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

p1 = 100
p2 = 100
mindist = 100

def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_prc(image):
    img_1_blur = cv2.GaussianBlur(image, (5, 5), 1)    # Gaussian Сгладить
    img_gray = cv2.cvtColor(img_1_blur, cv2.COLOR_BGR2GRAY)  # Преобразовать изображение в полутоновое

    img_blur = cv2.blur(img_gray, (5, 5))  # Сгладить

    img_canny = cv2.Canny(img_blur, 50, 150)  # Применить детектор границ Canny

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(img_canny, cv2.MORPH_GRADIENT, kernel)   # Morphological Gradient
    cv_show('closing', closing)

    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    img_contours = cv2.drawContours(img_canny, contours, -1, (255, 255, 255), 2)  # draw contours
    cv_show('conts', img_contours)

    length = len(contours)
    # Залить замкнутые области
    for i in range(0, length):
        cv2.fillPoly(img_contours, [contours[i]], (255, 255, 255))
    cv_show('res', img_contours)
    # return img_contours

def hough_search(image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, mindist, p1, p2, 10, 0)  # dp, minDist, param1_Canny threshold, param2 center threshold, minr, maxr
    circles = np.uint16(np.around(circles))

    count = 0
    r_true = []
    r_pred = []
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        r_true.append(int(i[2] / 10) * 10)
        r_pred.append(i[2])
        print(f"{count + 1}.({i[0]}, {i[1]}), R-detected: {i[2]}(pixel)")
        count += 1
        cv2.putText(img, f'{str(count)} radius={i[2] / 10}mm ', (i[0], i[1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)

    cv_show('final result', img)
    # return img, r_true, r_pred


if __name__ == "__main__":
    img = cv2.imread('for_search.jpg')
    img_contours = image_prc(img)
    # res = hough_search(img_contours)
    # res, y_true, y_pred = hough_search(img_contours)
    # cv_show(res)
    cv2.namedWindow('final result')
    cv2.createTrackbar('p1', 'final result', 1, 255, hough_search)
    cv2.createTrackbar('p2', 'final result', 1, 255, hough_search)
    cv2.createTrackbar('mindist', 'final result', 1, 200, hough_search)

    while 1:
        k = cv2.waitKey(1) & 0xff
        if k ==27:
            break
        p1 = cv2.getTrackbarPos('p1', 'final result')
        p2 = cv2.getTrackbarPos('p2', 'final result')
        mindist = cv2.getTrackbarPos('mindist', 'final result')

    cv2.destroyAllWindows()

    # print(y_true)
    # print(y_pred)
    # accuracy = accuracy_score(y_true, y_pred, normalize=True)
    # print(accuracy)
    # f_score = f1_score(y_true, y_pred, average="macro")
    # print(f"f_score: {f_score}")







