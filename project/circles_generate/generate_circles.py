import cv2
import numpy as np
import random
from circle_search.circle_search import cv_show

class RandomCircle(object):
    global circle_pos

    def __init__(self, r):
        self.radius = r  # radius
        self.img = self.create_img()  # generate image

    # Detect the distance between the center and the center of the circles
    def radius_detect(self, x, y):
        for pos in circle_pos:  # Traversing the central coordinates of the circle
            if ((pos[0] - x) ** 2 + (pos[1] - y) ** 2) ** 0.5 >= self.radius + pos[2]:   # Compare the distance of two points with the radius*2
                continue
            else:
                return 1
        return 0

    def create_img(self):
        # global img
        global circle_pos
        background = background_720p  # wood_background
        x = np.random.randint(self.radius, background.shape[1] - self.radius)  # circles should inside the border
        y = np.random.randint(self.radius, background.shape[0] - self.radius)
        while self.radius_detect(x, y):
            x = np.random.randint(self.radius, background.shape[1] - self.radius)
            y = np.random.randint(self.radius, background.shape[0] - self.radius)
        circle_pos.append((x, y, self.radius))  # add coord into list
        img = cv2.circle(background, (x, y), self.radius, (255, 255, 255), -1)  # draw circles (BGR)
        np.savetxt("circles_pos.txt", circle_pos, fmt='%d', delimiter=',')
        return img

def img_bitwise(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img_gray, 175, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(bg_circle, bg_circle, mask=mask)
    img_fg = cv2.bitwise_and(img, img, mask=mask_inv)
    dst = cv2.add(img_bg, img_fg)
    return dst

def data_save(name, list):
    list.sort()
    f = open(name, mode='w')
    dets = np.array(list)
    for i in dets:
        print(int(i), end=' ', file=f)

if __name__ == "__main__":
    background_720p = cv2.imread('/home/kehua/PycharmProjects/KP/project/image_size/720p.jpg')
    bg_circle = cv2.imread('BG.jpg')
    circle_pos = []
    list_giameter = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]   # ???????????????????????? ???????????? ??????????????????: 2, 3, 5, 7, 9, 10, 11, 12, 14, 16, 18, 20
    list_random_diameter = []
    idxs = np.random.randint(0, len(list_giameter), size=random.randint(15, 20))    # 4 - 6 random holes
    for i in idxs:
        list_random_diameter.append(
            list_giameter[i])

    for radius in list_random_diameter:
        rand = RandomCircle(r=radius)

    img = rand.img
    #cv2.imwrite('/home/kehua/PycharmProjects/KP/project/circle_search/for_search.jpg', img)
    dst = img_bitwise(img)
    cv_show('res', dst)
    cv2.imwrite("/home/kehua/PycharmProjects/KP/project/circle_search/for_search.jpg", dst)


    name = 'diameter_true'
    data_save(name, list_random_diameter)