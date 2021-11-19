# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 17:54:05 2020

@author: Administrator
"""

#----------------------绘图与鼠标操作Start--------------
#实例1：显示图片，在图片上点击左键显示点击点的坐标，
#点右键则显示点击点的三个颜色值。
import cv2

img = cv2.imread('strawberry.jpg', cv2.IMREAD_COLOR)

#定义回调函数
def click_(event, x, y, flags, param):
    #按下左键
    if event == cv2.EVENT_LBUTTONDOWN:
        #图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细(-1表示实心)
        cv2.putText(img, str((x,y)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) 
    #按下右键    
    elif event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y,x,0]
        green = img[y,x,1]
        red = img[y,x,2]
        cv2.putText(img, str((blue, green, red)), (x,y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (int(blue),int(green),int(red)), 2)
    cv2.imshow('IMG', img)

cv2.imshow('IMG', img)
#设置鼠标操作回调响应
cv2.setMouseCallback('IMG', click_)
cv2.waitKey(0)
cv2.destroyAllWindows()
   
 

#实例2：用鼠标画矩形
import cv2
import numpy as np

isDrawing = False
x0,y0 = -1, -1

def draw_rect(event, x, y, flags, param):
    global x0,y0,isDrawing
    if event == cv2.EVENT_LBUTTONDOWN:
        x0,y0 = x,y
        isDrawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDrawing == True:
            tmp = img.copy()
            cv2.rectangle(tmp, (x0,y0), (x,y), (255,0,0), 2)
            cv2.imshow('IMGWindow', tmp)
    elif event == cv2.EVENT_LBUTTONUP:
        isDrawing = False

img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('IMGWindow')
cv2.setMouseCallback('IMGWindow', draw_rect)
cv2.imshow('IMGWindow', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
      

##鼠标移动开始画圆，曲线由很多小圆组成，鼠标抬起停止画园
drawing = False

def draw_circle(event, x, y, flags, param):
    global drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True 
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x,y), 15, (0,0,255), 5)
        
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x,y), 15, (0,0,255), 5)   
        
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while(1):
    cv2.imshow('image', img)
    if cv2.waitKey(20)&0xFF == 27:
        break
cv2.destroyAllWindows()


#鼠标移动开始画圆，曲线由很多小圆组成，鼠标抬起停止画园
#增加画矩形，通过键入‘m’键切换

drawing = False
ix, iy = -1, -1
mode = True #默认画圆

def draw_circle(event, x, y, flags, param):
    global drawing, ix, iy, mode
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True 
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE: 
        if drawing == True:
            if mode == True:
                # 绘制圆圈，小圆点连在一起就成了线，3 代表了笔画的粗细
                cv2.circle(img, (x,y), 15, (0,0,255), 5)
            else:
                cv2.rectangle(img, (ix,iy), (x,y), (0,255,255), -1)
        
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.circle(img, (x,y), 15, (0,0,255), 5)
        else:
            cv2.rectangle(img, (ix,iy), (x,y), (0,255,255), -1)
            
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(20)&0xFF
    if k == 27:   
        break
    elif k == ord('m'):
        mode = not mode
cv2.destroyAllWindows()

#----------------------绘图与鼠标操作End--------------


#--------------------人脸检测Start-------------------

#人脸检测模型应用—图片人脸检测
from cv2 import cv2

#读取待处理图像
img = cv2.imread('facedetect2.jpg', cv2.IMREAD_COLOR)

#加载正面人脸检测分类器
#face_data = cv2.CascadeClassifier('C:\\Users\\Administrator\\Anaconda3\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
#加载正面人脸检测分类器
face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#检测人脸
faces = face_data.detectMultiScale(img, 1.3, 5)
#根据返回绘制人脸矩形框
for x,y,w,h in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),1)
#显示图像
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#人脸检测模型应用—视频人脸检测
import cv2

face_data = cv2.CascadeClassifier('C:\\Users\\Administrator\\Anaconda3\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_data.detectMultiScale(gray, 1.3, 5)
    for face in faces:
        x,y,w,h = face
        cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),1)
    cv2.imshow('face', img)
    flag = cv2.waitKey(1)
    if flag == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


#dlib人脸检测 dlibw未安装成功
#import cv2
#import dlib
#
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#
#cam = cv2.VideoCapture(0)
#
#while True:
#    _, img = cam.read()
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    dets = detector(gray, 1)
#
#    for k, d in enumerate(dets):
#        shape = predictor(img,d)
#        for i in range(68):
#            cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0,255,0),-1, 8)
#            cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, .5 , (255,0,0),1)
#
#    cv2.imshow("face", img)
#    flag = cv2.waitKey(1)
#    if flag == ord('q'):
#        break
#
#cam.release()
#cv2.destroyAllWindows()
#--------------------人脸检测End---------------------

#--------------------人脸识别Start-------------------
#第一步通过视频创建人脸文件
import cv2

#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('news.mp4')
face_date = cv2.CascadeClassifier('C:\\Users\\Administrator\\Anaconda3\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
num = 1
name_id = input("请输入id:")

while cam:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_date.detectMultiScale(gray, 1.3, 5)
    print(faces)

    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y),(x+w, y+h),(0,255,0),2)
    cv2.imshow('img',img)

    flag = cv2.waitKey(1)
    if flag == ord('q'):
        break
    if flag == ord('x'):#点一次X拍一张照片
        if len(faces) == 1 and num < 50:#人脸数限制
            #保存灰度照片，先在工作路径下手动创建face_img文件夹
            cv2.imwrite(r"./face_img/user_%s_%s.jpg"%(name_id, num), gray)
            num += 1
        else:
            print("未检测到人脸！")

cam.release()
cv2.destroyAllWindows()

#第二步训练人脸识别模型
import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('C:\\Users\\Administrator\\Anaconda3\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
def get_img_and_id(path):
    id = []
    face_data = []
    image_path = [os.path.join(path, i) for i in os.listdir(path)]
    #print(image_path)
    for item in image_path:
        image = Image.open(item).convert('L')
        image_np = np.array(image, 'uint8')
        #print(image_np)
        image_id = int(os.path.split(item)[1].split('.')[0].split('_')[1])
        faces = detector.detectMultiScale(image_np)
        #print(faces)
        for x,y,w,h in faces:
            #print(image_np[y:y+h, x:x+w])
            face_data.append(image_np[y:y+h, x:x+w])
            id.append(image_id)

    return face_data,id

faces, id = get_img_and_id(r'./face_img')
recognizer.train(faces, np.array(id))
recognizer.save(r'./trainer/trainer.yml')

#第三步 采用训练模型识别视频人脸
import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier('C:\\Users\\Administrator\\Anaconda3\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('news.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.2, 5)

    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),(111,111,111),2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        conf = 100 - conf

        if conf < 40:
            cv2.putText(img, str('unknown'), (x,y+10), font, 0.5,(0,255,0),1)
        else:
            cv2.putText(img, str(id),(x,y+10),font, 0.5,(0,255,0),1)
            cv2.putText(img,str(conf), (x+50, y+10),font, 0.5,(0,255,0),1)
    cv2.imshow('person', img)
    flag = cv2.waitKey(1)
    if flag == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
#--------------------人脸识别End---------------------
#批量修改文件
#import os
#path = r'D:\Learning\Programm\Python\MyOpenCV\face_img'
#filelist = os.listdir(path)
#for index, filename in enumerate(filelist, 1):
#    os.rename(path+'/user_George_%d.jpg'%(index), path+'/user_002_%d.jpg'%(index))

#--------------------视频操作Start---------------------
# 打开摄像头并灰度化显示
import cv2
capture = cv2.VideoCapture(0)
while(True):
    ret, frame = capture.read() # 获取一帧
    # 将这帧转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()



# 获取捕获的分辨率
# propId可以直接写数字，也可以用OpenCV的符号表示
import cv2
capture = cv2.VideoCapture(0)
width, height = capture.get(3), capture.get(4)
print(width, height)
capture.release()


#播放本地视频
import cv2
capture = cv2.VideoCapture('news.mp4')
while(capture.isOpened()):
    ret, q = capture.read()
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    if cv2.waitKey(30) == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()


#录制视频
import cv2
capture = cv2.VideoCapture(0)
# 定义编码方式并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#创建一个VideoWriter
outfile = cv2.VideoWriter('outputNew.avi', fourcc, 25.0,  (640, 480))
while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        outfile.write(frame)  # 一帧一帧写入
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
    
capture.release()
outfile.release()
cv2.destroyAllWindows()
#--------------------视频操作End---------------------