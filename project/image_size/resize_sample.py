import cv2

img = cv2.imread('/home/kehua/PycharmProjects/KP/project/colors/ясень.jpg')
img_resize = cv2.resize(img, (1280, 720))

cv2.imshow('res', img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('720pr.jpg', img_resize)
print(img_resize.shape)