import cv2

img = cv2.imread('originwood.jpg')
img_resize = cv2.resize(img, (960, 540))

cv2.imshow('res', img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('540p.jpg', img_resize)
print(img_resize.shape)