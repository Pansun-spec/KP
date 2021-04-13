import cv2
import numpy as np
import math

def cv_show(image):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_prc(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Преобразовать изображение в полутоновое
    cv_show(img_gray)

    img_blur = cv2.blur(img_gray, (3, 3))  # Сгладить

    img_canny = cv2.Canny(img_blur, 50, 150)  # Применить детектор границ Canny
    cv_show(img_canny)

    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    img_contours = cv2.drawContours(img_canny, contours, -1, (255, 255, 255), 2)  # draw contours

    cv_show(img_contours)

    length = len(contours)
    # Залить замкнутые области
    for i in range(0, length):
        cv2.fillPoly(img_contours, [contours[i]], (255, 255, 255))
    cv_show(img_contours)
    return img_contours

def hough_search(image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 100, 200, 30, 15, 0)
    circles = np.uint16(np.around(circles))

    count = 0
    pi = 3.14
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        print(f"{count + 1}.x: {i[0]}, y: {i[1]}, area: {i[2] ** 2 * pi}")
        count += 1
        if i[2] < 100:
            cv2.putText(img, f'{str(count)} R={math.ceil(i[2] / 10) * 10}mm', (i[0], i[1]), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(img, f'{str(count)} R={math.ceil(i[2] / 100) * 100}mm', (i[0], i[1]), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

    cv_show(img)


if __name__ == "__main__":
    img = cv2.imread('for_search.jpg')
    img_contours = image_prc(img)
    hough_search(img_contours)







