# trainer.py
# Train recognizer using captured face images
#
# Project: Face Recognition using OpenCV and Raspberry Pi
# Ref: https://www.pytorials.com/face-recognition-using-opencv-part-2/
# By: Mickey Chan @ 2019

# Import required modules
import os
import numpy as np 
from PIL import Image 
import cv2
import time

# Setup Classifer and create Recognizer
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#載入人臉檢測器（分類器）
recognizer = cv2.face.LBPHFaceRecognizer_create() # or LBPHFaceRecognizer_create()
#載入人臉識別器

# Create directory for storing trained data
baseDir = os.path.dirname(os.path.abspath(__file__))
imageDir = os.path.join(baseDir, "dataset")
recognizerDir = os.path.join(baseDir, "recognizer")

if not os.path.exists(recognizerDir):
    os.makedirs(recognizerDir)

# Dataset of face data for training
yIDs = []
xFaces = []
#放置資料庫內容

# Walk through all captured face data
beginTime = time.time()
for root, dirs, files in os.walk(imageDir):
    print(root, dirs, files)
    for file in files:
        print(file)
        
        # Only process PNG and JPEG images
        if file.endswith("png") or file.endswith("jpg"):   #只抓取圖片檔
            # Retrieve USER ID from directory name
            path = os.path.join(root, file)    #起始路徑
            id_ = int(os.path.basename(root))  #取出路徑中的檔案名稱
            print("UID:" + str(id_))
            
            # Convert the face image to grayscale and convert pixel data to Numpy Array
            faceImage = Image.open(path).convert("L")  #使用函式convert()來進行轉換L (8-bit pixels, black and white)
            faceArray = np.array(faceImage, "uint8") #讀取照片陣列
            
            # Insert USER ID and face data into dataset
            yIDs.append(id_)          #檔案名稱放進yIDS
            xFaces.append(faceArray)  #讀取照片陣列放進xFaces
            
            # Display the face image to be used for training
            cv2.imshow("training", faceArray)
            cv2.waitKey(10)

# Train recognizer and then save trained model
recognizer.train(xFaces, np.array(yIDs))     #將ID與face導入訓練模組
recognizer.save(recognizerDir + "/trainingData.yml")    #將訓練檔案存檔

# Clean up
cv2.destroyAllWindows()
print("DONE")
elapsedTime = round(time.time() - beginTime, 4)
print("Elapsed time: " + str(elapsedTime) + "s")
