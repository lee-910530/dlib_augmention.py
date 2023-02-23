import cv2
import numpy as np
from matplotlib import pyplot as plt
import dlib
import imutils
from PIL import ImageGrab
import pyautogui as pag


def threshold_way() :
   img = cv2.imread('color.jpeg', 0)
   ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
   ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
   ret, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
   ret, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
   ret, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

   titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
   images = [img, th1, th2, th3, th4, th5]

   for i in range(6):
      plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
      plt.title(titles[i])
      plt.xticks([]), plt.yticks([])
   plt.show()

def Averaging():
   img = cv2.imread('color.jpeg')
   blur = cv2.blur(img, (10, 10))

   plt.subplot(121), plt.imshow(img[:,:,[2,1,0]]), plt.title('Image')
   plt.xticks([]), plt.yticks([])
   plt.subplot(122), plt.imshow(blur[:,:,[2,1,0]]), plt.title('Blur')
   plt.xticks([]), plt.yticks([])
   plt.show()

def Gaussian_Filter():
   img = cv2.imread('color.jpeg')

   gauss = cv2.GaussianBlur(img, (3, 3), 0)

   plt.subplot(121), plt.imshow(img[:,:,[2,1,0]]), plt.title('Image')
   plt.xticks([]), plt.yticks([])
   plt.subplot(122), plt.imshow(gauss[:,:,[2,1,0]]), plt.title('Gauss')
   plt.xticks([]), plt.yticks([])
   plt.show()

def draw_circle_or_line():
   image = np.zeros((480, 640, 3), np.uint8)

   image.fill(128)

   color = (255, 0, 0)
   # 注意三個參數對應的不是RGB而是BGR
   cv2.line(image, (0, 0), (255, 255), color, 5)
   # 意思是在img的底圖上，繪製一條從矩陣左上角(0, 0)到矩陣右下角(255, 255)的紅色線條，
   # 寬度設定為正5
   cv2.circle(image, (300, 200), 50, color, -1)
   # 參數分別為：影像、圓心座標、半徑、顏色和線條寬度。線條寬度如果設定為正值，
   # 則代表正常的線條寬度，設定為負值，則代表畫實心的圓圈。

   cv2.imshow('Result', image)
   cv2.waitKey(0)

def draw_text():
   img = np.zeros((400, 400, 3), np.uint8)
   img.fill(128)

   text = "I'm here!"

   cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
   cv2.putText(img, text, (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1, cv2.LINE_AA)
   cv2.putText(img, text, (10, 120), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
   cv2.putText(img, text, (10, 160), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
   cv2.putText(img, text, (10, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA)
   cv2.putText(img, text, (10, 240), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 1, cv2.LINE_AA)
   cv2.putText(img, text, (10, 280), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
   cv2.putText(img, text, (10, 320), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA)

   cv2.imshow('TextImage', img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

def draw_elipse():
   img = np.zeros((400, 400, 3), np.uint8)
   img.fill(128)

   # 中心點
   center = (300, 300)
   # 軸長
   axes = (100, 50)
   # 角度
   angle = 45
   # 起始角度
   startAngle = 0
   # 結束角度
   endAngle = 270
   # 顏色
   color = (0, 255, 0)
   # 寬度
   thickness = 1

   cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)

   cv2.imshow('ellipseImage', img)

   cv2.waitKey(0)
   cv2.destroyAllWindows()

def draw_polylines():
   img = np.zeros((400, 400, 3), np.uint8)
   img.fill(128)
   # 頂點座標
   pt = np.array([[200, 200], [300, 100], [400, 200], [400, 400], [200, 400]], np.int32)

   # 用reshape轉為(-1, 1, 2)的陣列
   pt = pt.reshape((-1, 1, 2))

   # 顏色
   color = (0, 255, 0)

   # cv2.polylines(影像, 頂點座標, 封閉線, 顏色, 寬度)
   cv2.polylines(img, [pt], False, color, 2)

   cv2.imshow('polylineImage', img)

   cv2.waitKey(0)
   cv2.destroyAllWindows()

def draw_fillPoly():
   img = np.zeros((400, 400, 3), np.uint8)
   img.fill(128)
   # 頂點座標
   pt = np.array([[200, 200], [300, 100], [400, 200], [400, 400], [200, 400]], np.int32)

   # 用reshape轉為(-1, 1, 2)的陣列
   pt = pt.reshape((-1, 1, 2))

   # 顏色
   color = (0, 255, 0)

   # cv2.fillPoly(影像, 頂點座標, 顏色)
   cv2.fillPoly(img, [pt], color)

   cv2.imshow('fillPolyImage', img)

   cv2.waitKey(0)
   cv2.destroyAllWindows()

def x():
   img = cv2.imread('humans.jfif')

   # 縮小圖片
   img = imutils.resize(img, width=1280)

   # Dlib 的人臉偵測器
   detector = dlib.get_frontal_face_detector()

   # 偵測人臉
   face_rects = detector(img, 0)

   # 取出所有偵測的結果
   for i, d in enumerate(face_rects):
      x1 = d.left()
      y1 = d.top()
      x2 = d.right()
      y2 = d.bottom()

      # 以方框標示偵測的人臉
      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

   # 顯示結果
   cv2.imshow("Face Detection", img)

   cv2.waitKey(0)
   cv2.destroyAllWindows()

   print(dlib.DLIB_USE_CUDA)

def y():
   cap = cv2.VideoCapture(0)

   # 取得畫面尺寸
   width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

   # 使用 XVID 編碼
   fourcc = cv2.VideoWriter_fourcc(*'XVID')

   # 建立 VideoWriter 物件，輸出影片至 output.avi，FPS 值為 20.0
   out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

   # Dlib 的人臉偵測器
   detector = dlib.get_frontal_face_detector()

   # 以迴圈從影片檔案讀取影格，並顯示出來
   while (cap.isOpened()):
      ret, frame = cap.read()

      # 偵測人臉
      face_rects, scores, idx = detector.run(frame, 0)

      # 取出所有偵測的結果
      for i, d in enumerate(face_rects):
         x1 = d.left()
         y1 = d.top()
         x2 = d.right()
         y2 = d.bottom()
         text = "%2.2f(%d)" % (scores[i], idx[i])

         # 以方框標示偵測的人臉
         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

         # 標示分數
         cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
                     0.7, (255, 255, 255), 1, cv2.LINE_AA)

      # 寫入影格
      out.write(frame)

      # 顯示結果
      cv2.imshow("Face Detection", frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
         break

   cap.release()
   out.release()
   cv2.destroyAllWindows()

def z():
   image = ImageGrab.grab()
   width, height = image.size

   fourcc = cv2.VideoWriter_fourcc(*'XVID')

   detector = dlib.get_frontal_face_detector()
   # video = cv2.VideoWriter('test.avi', fourcc, 25, (width, height))

   while True:

      img_rgb = ImageGrab.grab()
      img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)


      # video.write(img_bgr)

      cv2.imshow('imm', img_bgr)

      if cv2.waitKey(1) & 0xFF == ord('0'):
         break

   # video.release()
   cv2.destroyAllWindows()

def test():
   while 1 :
      print(pag.position())
      img_rgb = ImageGrab.grab((0,0,1280,1600))

      img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

      # detector = dlib.get_frontal_face_detector()
      detector = dlib.simple_object_detector(r"C:\Program Files\Python310\dlib-19.24\dlib-19.24\tools\imglab\build\Release\butterfly.svm")

      # 偵測人臉
      face_rects = detector(img_bgr, 0)
      cv2.namedWindow("imm", 0)
      cv2.resizeWindow("imm", 600,600)
      # 取出所有偵測的結果
      for i, d in enumerate(face_rects):
         x1 = d.left()
         y1 = d.top()
         x2 = d.right()
         y2 = d.bottom()
         print(x1,y1)

         # 以方框標示偵測的人臉
         cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)


      cv2.imshow('imm', img_bgr)

      if cv2.waitKey(1) & 0xFF == ord('0'):
         break

   cv2.destroyAllWindows()
if __name__ == '__main__':
   pass

