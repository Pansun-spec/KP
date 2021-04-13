import cv2
import numpy as np
import random
import imutils
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dis

def img_resize(image):
    height = image.shape[0]
    width = image.shape[1]
    width_new = 1280
    height_new = 720

    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))

    return img_new

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
        background = background_1080p  # wood_background
        x = np.random.randint(self.radius, background.shape[1] - self.radius)  # circles should inside the border
        y = np.random.randint(self.radius, background.shape[0] - self.radius)
        while self.radius_detect(x, y):
            x = np.random.randint(self.radius, background.shape[1] - self.radius)
            y = np.random.randint(self.radius, background.shape[0] - self.radius)
        circle_pos.append((x, y, self.radius))  # add coord into list
        img = cv2.circle(background, (x, y), self.radius, (18, 74, 115), -1)  # draw circles (BGR)
        return img


if __name__ == "__main__":
    background_1080p = cv2.imread('1080p.jpg')
    circle_pos = []
    list_radius = [20, 30, 50, 70, 90, 100, 110, 120, 140, 160, 180, 200]   # Используются сверла диаметром: 2, 3, 5, 7, 9, 10, 11, 12, 14, 16, 18, 20.
    list_random_radius = []
    idxs = np.random.randint(0, len(list_radius), size=random.randint(4, 6))    # 4 - 6 random holes
    for i in idxs:
        list_random_radius.append(list_radius[i])
    # print(idxs[:])
    # radius = random.choice(list_radius)
    for radius in list_random_radius:
        rand = RandomCircle(r=radius)
    img = rand.img
    img_new = img_resize(rand.img)

    cv2.imshow("circle", img_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("E://PycharmProjects//KP//project//circle_search//for_search.jpg", img_new)

