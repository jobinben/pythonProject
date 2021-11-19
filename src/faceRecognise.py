# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:54:53 2020

@author: Administrator
"""

#--------------------人脸识别Start-------------------
#第一步通过视频创建人脸文件
import cv2
import cv2
import os

path_img = './face_img'
if not os.path.exists(path_img):
    os.mkdir(path_img)
    
cam = cv2.VideoCapture('news.mp4')#视频
#cam = cv2.VideoCapture(0)#摄像头

#face_date = cv2.CascadeClassifier('C:\\Users\\Administrator\\Anaconda3\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
face_date = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

num = 1
name_id = input("请输入id:")

while(cam.isOpened()):
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
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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

path_train = './trainer'
if not os.path.exists(path_train):
    os.mkdir(paht_train)
recognizer.save(r'./trainer/trainer.yml')

#第三步 采用训练模型识别视频人脸
import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture('news2.mp4')
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