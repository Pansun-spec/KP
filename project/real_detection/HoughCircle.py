import sys
import numpy as np
import cv2 as cv
from skimage.io import imread, imsave


def cv_show(name, image):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main(argv):
    default_file = 'for_search.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    #Загрузка картинки
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    #Проверка источника
    if src is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1

    #Перевод в градацию серого:
    gray_img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray_img = cv.medianBlur(gray_img, 5)   #Применение срединного размытия, чтобы уменьшить шум и избежать ложного обнаружения окружности:
    cv_show('gray_image', gray_img)

    rows = gray_img.shape[0]    #количество рядов
    canny_img = cv.Canny(gray_img, 50, 150) #применение детектора границ кенни
    cv_show('canny_image', canny_img)

    # Применение алгоритма Хафа
    # - gray: Входное изображение (в градации серого).
    # - circles: Вектор из трех значений: xc,yc,r для каждого найденного круга.
    # - HOUGH_GRADIENT: Метод обнаружения
    # - dp = 1: Обратное соотношение разрешения.
    # - min_dist = gray.rows/16: Минимальное расстояние между координатами центра (x, y) обнаруженных окружностей.
    # Если min_dist слишком мало, может быть (ошибочно) обнаружено несколько кругов в том же районе, что и оригинал.
    # Если min_dist слишком велико, то некоторые круги могут вообще не обнаруживаться.
    # - param_1 = 200: Верхний порог для внутреннего детектора Canny.
    # - param_2 = 100*: Пороговое значение накопителя для метода cv.HOUGH_GRADIENT.
    # Чем меньше пороговое значение, тем больше кругов будет обнаружено (включая ложные круги).
    # Чем больше порог, тем больше кругов потенциально будет возвращено.
    # - min_radius = 0: Минимальный радиус кругов
    # - max_radius = 0: Максимальный радиус кругов
    # большее количество определенных кругов получается путем оптимального подбора параметров
    # меняем параметр расстояния , пороговые значения алгоритма Кенни и накопителя градиента,
    # также явно прописывам минимальный и максимальный радиусы
    circles = cv.HoughCircles(gray_img, cv.HOUGH_GRADIENT, 1, rows/32,
                              param1=80, param2=20,
                              minRadius=2, maxRadius=50)

    if circles is not None:

        circles = np.uint16(np.around(circles))
        count = 0
        for i in circles[0, :]:
            center = (i[0], i[1])
            # центр круга
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # окружность
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
            print(f"{count + 1}.({i[0]}, {i[1]}), diameter: {round(i[2] / 5)}")
            count += 1
            # вывод текста
            # cv.putText(src, f'{str(count)} radius={int(i[2] /10 / 2 * 10)}', (i[0], i[1]),
            #       cv.FONT_HERSHEY_COMPLEX, 4, (0, 0, 255), 2)

    cv_show('detected circles', src)
    imsave('res_normalHough', src)
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])