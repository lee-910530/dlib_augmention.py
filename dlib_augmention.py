import cv2
import numpy as np
from matplotlib import pyplot as plt
import dlib
import imutils
import math
from PIL import ImageGrab
import pyautogui as pag
import os
import random

def rotate(origin, point, angle):
    angle = math.radians(-angle)

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return int(qx), int(qy)

def randrange(a, b):
    return random.randrange(a, b)

class Image_argue():
    def __init__(self):


        self.blur_size = 1
        self.rotation_angle = 0



    def train_parameters(self,file_name):
        title = []
        place = []
        name = ""
        f = open(file_name, "r")
        word = f.readlines()

        for i in word:

            if i[0:14] == "  <image file=":
                x = i.replace("  <image file=", "").replace("\\", '\' ').replace(" width=", '').replace(" height=", '').replace(">\n", '').split("'")

                y = list(filter(None, x))
                title.append(y)
            elif i[0:13] == "    <box top=":
                x = i.replace("    <box top=", "").replace(" left=", '').replace(" width=",'').replace(
                    " height=", '').replace("/>\n", '').split("'")
                y = list(filter(None, x))
                z = list(map(int,y))
                place.append(z)

        return(title,place)

    def blur_argue(self,implement,image):
        if implement == True:
            self.blur_size = randrange(min_blur_size, max_blur_size)
            if self.blur_size % 2 == 0:
                self.blur_size = self.blur_size + 1
            blur_after = cv2.GaussianBlur(image, (self.blur_size, self.blur_size), 0)
            place[i][0] = place[i][0]
            place[i][1] = place[i][1]
            place[i][2] = place[i][2]
            place[i][3] = place[i][3]
            return blur_after

        elif implement == False :
            blur_after = image
            return blur_after



    def rotate_argue(self,implement,image):
        if implement == True:
            height, width = image.shape[:2]
            print(height, width)

            angle = randrange(min_rotation_angle, max_rotation_angle)
            rotated_after = imutils.rotate(image, angle)

            origin = (width / 2, height / 2)

            a = place[i][1] + (place[i][2])/2
            b = place[i][0] + place[i][3]/2

            a,b = rotate(origin,(a,b),angle)

            place[i][1] = int(a - place[i][2]/2)
            place[i][0] = int(b - place[i][3]/2)
            place[i][2] = int(place[i][2])
            place[i][3] = int(place[i][3])
            return rotated_after

        elif implement == False :
            rotate_after =image
            return rotate_after











img_arg = Image_argue()

if __name__ == '__main__':
    file = "1.xml"
    image_cnt = 5
    continue_cnt = 1
    name = "butterfly\\image_" + str(image_cnt).rjust(4,"0") + ".png"

    max_blur_size = 15
    min_blur_size = 1
    max_rotation_angle = 10
    min_rotation_angle = -10

    blur_sw = True
    rotate_sw = True


    title, place = img_arg.train_parameters(file)
    print(title)
    print(place)
    for i in range(len(title)):

        img = cv2.imread(title[0][0] + "/" + "image_" + str(i+1).rjust(4,"0") + ".jpg")


        rotate_after = img_arg.rotate_argue(rotate_sw, img)
        blur_after = img_arg.blur_argue(blur_sw,rotate_after)

        cv2.rectangle(blur_after, (place[i][1], place[i][0]), (place[i][2]+place[i][1], place[i][0]+place[i][3]), (0, 255, 0), 4, cv2.LINE_AA)





        cv2.imshow("after",blur_after)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # ================================================================



    # gauss = cv2.GaussianBlur(img, (img_arg.blur_size, img_arg.blur_size), 0)




    # color = (255, 0, 0)
    # # 注意三個參數對應的不是RGB而是BGR
    # cv2.rectangle(image, (34,44), (247+34, 171+44), color, 5)


    # cv2.imshow('Result', gauss)
    # cv2.waitKey(0)
    #
    # for i in range(len(title)):
    #     img_arg.blur_argue(blur_sw)
    #     print(i)
    #     img =cv2.imread( "butterfly\\image_" + str(i+1).rjust(4,"0") + ".png")
    #     # gauss = cv2.GaussianBlur(img, (img_arg.blur_size, img_arg.blur_size), 0)
    #
    #     cv2.imshow("after",img)
    #     cv2.waitKey(0)


