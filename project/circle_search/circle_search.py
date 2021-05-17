import cv2
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

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
    img_1_blur = cv2.GaussianBlur(image, (5, 5), 1)    # Gaussian Сгладить
    img_gray = cv2.cvtColor(img_1_blur, cv2.COLOR_BGR2GRAY)  # Преобразовать изображение в полутоновое

    img_blur = cv2.blur(img_gray, (5, 5))  # Сгладить

    img_canny = cv2.Canny(img_blur, 50, 150)  # Применить детектор границ Canny

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(img_canny, cv2.MORPH_GRADIENT, kernel)   # Morphological Gradient
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
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 100, 255, 255, 17, 0)  # dp, minDist, param1_Canny threshold, param2 center threshold, minr, maxr
    circles = np.uint16(np.around(circles))

    count = 0
    r_pred = []
    for i in circles[0, :]:
        # if i[2] > 100 or i[2] < 10:
        #     continue
        # else:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        r_pred.append(round(i[2] / 10) * 10)
        r_pred.sort()
        print(f"{count + 1}.({i[0]}, {i[1]}), d-detected: {i[2]}(pixel)")
        count += 1
        cv2.putText(img, f'{str(count)} radius={int(round(i[2] / 10) / 2 * 10)}mm ', (i[0], i[1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)

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

if __name__ == "__main__":
    img = cv2.imread('for_search.jpg')

    img_contours = image_prc(img)
    pred_lose = hough_search(img_contours)
    print(f"pred_lose: {pred_lose}")
    path = "E:/PycharmProjects/KP/project/circles_generate/radius_true"
    y_true = data_get(path)
    print(f"y_true: {y_true}")

    y_pred = list_proc(pred_lose, y_true)
    print(f"y_pred: {y_pred}")

    accuracy = accuracy_score(y_true, y_pred, normalize=True)
    print(f"accuracy: {accuracy}")
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print(f"f1-score: {f1_macro}")







