import cv2
import numpy as np
from matplotlib import pyplot as plt

import timeit
import dlib
import imutils
import math
from PIL import ImageGrab
import pyautogui as pag
import os
import random

def show_img(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()

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
        self.offset_x = 0
        self.offset_y = 0
        self.angle = 0

        self.blur_size = 1
        self.rotation_angle = 0

        self.lightness = 0
        self.saturation = 0

        self.intensity = 1

        self.contrast = 0

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

    def move_argue (self,implement,image):
        if implement == True:
            self.offset_x = randrange(min_move_offset_x, max_move_offset_x)
            self.offset_y = randrange(min_move_offset_y, max_move_offset_y)

            move_after = imutils.translate(image, self.offset_x, self.offset_y)

            a = place[i][1] + self.offset_x
            b = place[i][0] + self.offset_y


            place[i][1] = a
            place[i][0] = b

            place[i][0] = place[i][0]
            place[i][1] = place[i][1]
            place[i][2] = place[i][2]
            place[i][3] = place[i][3]
            return move_after

        elif implement == False:
            move_after = image
            return move_after

    def lightness_saturation_argue(self,implement,img):
        if implement == True:
            self.lightness = randrange(min_lightness, max_lightness)
            self.saturation = randrange(min_saturation, max_saturation)
            origin_img = img

            # 圖像歸一化，且轉換為浮點型
            fImg = img.astype(np.float32)
            fImg = fImg / 255.0

            # 顏色空間轉換 BGR -> HLS
            hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
            hlsCopy = np.copy(hlsImg)

            #     lightness = 0 # lightness 調整為  "1 +/- 幾 %"
            #     saturation = 300 # saturation 調整為 "1 +/- 幾 %"

            # 亮度調整
            hlsCopy[:, :, 1] = (1 + self.lightness / 100.0) * hlsCopy[:, :, 1]
            hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1

            # 飽和度調整
            hlsCopy[:, :, 2] = (1 + self.saturation / 100.0) * hlsCopy[:, :, 2]
            hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1

            # 顏色空間反轉換 HLS -> BGR
            result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
            result_img = ((result_img * 255).astype(np.uint8))

            return result_img
        if implement == False:
            result_img = img
            return result_img

    def intensity_argue(self,implement,img):
        if implement == True:
            self.intensity = random.uniform(min_intensity, max_intensity)

            origin_img = img

            maxIntensity = 255.0  # depends on dtype of image data

        # Parameters for manipulating image data
            phi = 1
            theta = 1

            # Decrease intensity
            after_img = (maxIntensity / phi) * (origin_img / (maxIntensity / theta)) ** self.intensity
            after_img = np.array(after_img, dtype=np.uint8)
            return after_img

        elif implement == False :
            after_img = img
            return after_img

    def rotate_argue(self,implement,image):
        if implement == True:
            height, width = image.shape[:2]
            print(height, width)

            self.angle = randrange(min_rotation_angle, max_rotation_angle)
            rotated_after = imutils.rotate(image, self.angle)

            origin = (width / 2, height / 2)

            a = place[i][1] + (place[i][2])/2
            b = place[i][0] + place[i][3]/2

            a,b = rotate(origin,(a,b),self.angle)

            place[i][1] = int(a - place[i][2]/2)
            place[i][0] = int(b - place[i][3]/2)
            place[i][2] = int(place[i][2])
            place[i][3] = int(place[i][3])
            return rotated_after

        elif implement == False :
            rotate_after =image
            return rotate_after

    def modify_color_temperature(self,img):

        # ---------------- 冷色調 ---------------- #

        #     height = img.shape[0]
        #     width = img.shape[1]
        #     dst = np.zeros(img.shape, img.dtype)

        # 1.計算三個通道的平均值，並依照平均值調整色調
        imgB = img[:, :, 0]
        imgG = img[:, :, 1]
        imgR = img[:, :, 2]

        # 調整色調請調整這邊~~
        # 白平衡 -> 三個值變化相同
        # 冷色調(增加b分量) -> 除了b之外都增加
        # 暖色調(增加r分量) -> 除了r之外都增加
        bAve = cv2.mean(imgB)[0]
        gAve = cv2.mean(imgG)[0] + 20
        rAve = cv2.mean(imgR)[0] + 20
        aveGray = (int)(bAve + gAve + rAve) / 3

        # 2. 計算各通道增益係數，並使用此係數計算結果
        bCoef = aveGray / bAve
        gCoef = aveGray / gAve
        rCoef = aveGray / rAve
        imgB = np.floor((imgB * bCoef))  # 向下取整
        imgG = np.floor((imgG * gCoef))
        imgR = np.floor((imgR * rCoef))

        # 3. 變換後處理
        #     for i in range(0, height):
        #         for j in range(0, width):
        #             imgb = imgB[i, j]
        #             imgg = imgG[i, j]
        #             imgr = imgR[i, j]
        #             if imgb > 255:
        #                 imgb = 255
        #             if imgg > 255:
        #                 imgg = 255
        #             if imgr > 255:
        #                 imgr = 255
        #             dst[i, j] = (imgb, imgg, imgr)

        # 將原文第3部分的演算法做修改版，加快速度
        imgb = imgB
        imgb[imgb > 255] = 255

        imgg = imgG
        imgg[imgg > 255] = 255

        imgr = imgR
        imgr[imgr > 255] = 255

        cold_rgb = np.dstack((imgb, imgg, imgr)).astype(np.uint8)

        print("Cold color:")
        print(cold_rgb.shape)
        show_img(cold_rgb)

    def gaussian_noise(self,implement,img,gaussian_noise_z):
        if implement == True:
            # int -> float (標準化)
            img = img / 255
            # 隨機生成高斯 noise (float + float)
            noise = np.random.normal(0, gaussian_noise_z , img.shape)
            # noise + 原圖
            gaussian_out = img + noise
            # 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
            gaussian_out = np.clip(gaussian_out, 0, 1)

            # 原圖: float -> int (0~1 -> 0~255)
            gaussian_out = np.uint8(gaussian_out * 255)
            return gaussian_out
        elif implement == False:
            gaussian_out = img
            return gaussian_out

    def modify_contrast_and_brightness2(self,implement,img):
        if implement == True:
            # 上面做法的問題：有做到對比增強，白的的確更白了。
            # 但沒有實現「黑的更黑」的效果
            self.contrast= randrange(min_contrast, max_contrast)# - 減少對比度/+ 增加對比度 255 ~ -255
            brightness = 0


            B = brightness / 255.0
            c = self.contrast / 255.0
            k = math.tan((45 + 44 * c) / 180 * math.pi)

            img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)

            # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
            img = np.clip(img, 0, 255).astype(np.uint8)
            return img
        elif implement == False :
            return img

    def reduce_highlights(self,img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 先轉成灰階處理
        ret, thresh = cv2.threshold(img_gray, 200, 255, 0)  # 利用 threshold 過濾出高光的部分，目前設定高於 200 即為高光
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_zero = np.zeros(img.shape, dtype=np.uint8)

        #     print(len(contours))

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            img_zero[y:y + h, x:x + w] = 255
            mask = img_zero


        # alpha，beta 共同決定高光消除後的模糊程度
        # alpha: 亮度的缩放因子，默認是 0.2， 範圍[0, 2], 值越大，亮度越低
        # beta:  亮度缩放後加上的参数，默認是 0.4， 範圍[0, 2]，值越大，亮度越低
        result = cv2.illuminationChange(img, mask, alpha=2, beta=2)


        return result



img_arg = Image_argue()

if __name__ == '__main__':
    file = "1.xml"
    image_cnt = 5
    continue_cnt = 1
    name = "butterfly\\image_" + str(image_cnt).rjust(4,"0") + ".jpg"

    # modify_color_temperature 未使用
    # reduce_highlight 未使用

    max_blur_size = 9
    min_blur_size = 1
    # ================================================================
    max_rotation_angle = 10
    min_rotation_angle = -10
    # ================================================================
    max_move_offset_x = 20
    min_move_offset_x = -20
    max_move_offset_y = 20
    min_move_offset_y = -20
    # ================================================================
    min_lightness = -50
    max_lightness = 50
    min_saturation = -50
    max_saturation = 50
    # ================================================================
    min_intensity = 0.5
    max_intensity = 2
    # ================================================================
    gaussian_noise_range = 0.02  # 0~1
    # ================================================================
    min_contrast = -100
    max_contrast = 100  # 255 ~ -255

    blur_sw = False
    rotate_sw,move_sw = False,False
    lightness_saturation_sw = False
    intensity_sw = False
    gaussian_noise_sw = False
    contrast_and_brightness_sw = False




    img_temp =[]
    title, place = img_arg.train_parameters(file)

    for i in range(len(title)):

        img = cv2.imread(title[0][0] + "/" + "image_" + str(i+1).rjust(4,"0") + ".jpg")


        img_temp = img_arg.rotate_argue(rotate_sw, img)
        img = img_arg.move_argue(move_sw, img_temp)
        img_temp = img_arg.blur_argue(blur_sw,img)
        img = img_arg.lightness_saturation_argue(lightness_saturation_sw,img_temp)
        img_temp = img_arg.intensity_argue(intensity_sw,img)
        img = img_arg.gaussian_noise(gaussian_noise_sw,img_temp,gaussian_noise_range)
        img_temp = img_arg.modify_contrast_and_brightness2(contrast_and_brightness_sw,img)
        # img = img_arg.reduce_highlights(img_temp)



        img = img_temp
        cv2.rectangle(img, (place[i][1], place[i][0]), (place[i][2]+place[i][1], place[i][0]+place[i][3]), (0, 255, 0), 2, cv2.LINE_AA)
        print(place[i])

        print("butterfly\\image_" + str(image_cnt*continue_cnt+1+i).rjust(4, "0") + ".jpg")

        cv2.imshow("after",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




