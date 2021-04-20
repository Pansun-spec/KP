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
        return img

def img_bitwise(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img_gray, 175, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(bg_circle, bg_circle, mask=mask)
    img_fg = cv2.bitwise_and(img, img, mask=mask_inv)
    dst = cv2.add(img_bg, img_fg)
    return dst


if __name__ == "__main__":
    background_720p = cv2.imread('720p.jpg')
    bg_circle = cv2.imread('BG.jpg')
    circle_pos = []
    list_radius = [10, 15, 25, 35, 45, 50, 55, 60, 70, 80, 90, 100]   # Используются сверла диаметром: 2, 3, 5, 7, 9, 10, 11, 12, 14, 16, 18, 20
    list_random_radius = []
    idxs = np.random.randint(0, len(list_radius), size=random.randint(4, 6))    # 4 - 6 random holes
    for i in idxs:
        list_random_radius.append(
            list_radius[i])
    print(list_random_radius)
    # print(idxs[:])
    # radius = random.choice(list_radius)
    for radius in list_random_radius:
        rand = RandomCircle(r=radius)

    img = rand.img
    dst = img_bitwise(img)
    cv_show(dst)
    cv2.imwrite("E://PycharmProjects//KP//project//circle_search//for_search.jpg", dst)

