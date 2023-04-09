# Importing essential libraries
import numpy as np
import cv2

img = cv2.imread("22.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


blur = cv2.GaussianBlur(gray, (3, 3), 0)

#最小門檻值和最大門檻值，只有色調在門檻內的灰階像素能被辨識為圖片邊緣
edge = cv2.Canny(blur, 100, 240)


contours, _ = cv2.findContours(edge, cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

cnt = sorted(contours, key = cv2.contourArea)[-1]
# cv2.drawContours(img, [cnt], -1, (0,255,0), 5)

_, (w, h), angle = cv2.minAreaRect(cnt)
# w,h are floats, we would need to convert them to int first
w, h = int(w), int(h)

if angle > 45 or angle < -45:
    w,h = h,w
# the shape of cnt is (4,1,2) so for ease later on
# i will reshape it to (4,2)
cnt = cnt.reshape(cnt.shape[0], cnt.shape[-1])


# the four corners are the ones that have either extreme
# x-coordinate or extreme y-coordinate
s1 = sorted(cnt, key = lambda x : (x[0], x[1]))
s2 = sorted(cnt, key = lambda x : (x[1], x[0]))
corner1, corner3 = s1[0], s1[-1]
corner2, corner4 = s2[0], s2[-1]

corners = np.array([corner1, corner2, corner3, corner4])

target_corners = np.array([(0, 0), (w, 0), (w, h), (0, h)])

H, _ = cv2.findHomography(corners, target_corners, params=None)

transformed_image = cv2.warpPerspective(img, H,(img.shape[1],img.shape[0]))

# cropping the image out
transformed_image = transformed_image[:h,:w]

cv2.imshow("contours", transformed_image)

cv2.waitKey(0)

