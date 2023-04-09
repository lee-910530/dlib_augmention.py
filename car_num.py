import pytesseract
from PIL import Image
import cv2
import time
import datetime
import numpy as np

def timedata():
    loc_dt = datetime.datetime.today()
    loc_dt_format = loc_dt.strftime("%Y/%m/%d %H:%M:%S")
    return loc_dt_format



if __name__ == "__main__":
    cnt =0
    while  True:
        f = open(r".\car_num_data\car_num_cam.txt", "r")
        word = f.read(1)
        print(word)
        if word == "1" :
            cam = cv2.VideoCapture(0)

            while True:
                ret, img = cam.read()
                cv2.imwrite(r".\car_num_data\car_num_img\1.jpg",img)
                break

            cam.release()
            cv2.destroyAllWindows()

            pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            img = Image.open(r".\car_num_data\car_num_img\1.jpg")

            print(pytesseract.image_to_string(img, lang="eng"))
            f = open(".\car_num_data\car_num_information.txt", "a",)
            timenow = timedata()
            f.write(pytesseract.image_to_string(img, lang="eng") + timenow+"\n")

            time.sleep(2)
            f = open(".\car_num_data\car_num_cam.txt", "r+")
            f.write("2")


