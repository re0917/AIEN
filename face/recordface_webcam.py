# recordface_webcam.py
# Capture face image of a person for face recognition, using webcam or IP cam
#
# Project: Face Recognition using OpenCV and Raspberry Pi
# Ref: https://www.pytorials.com/face-recognition-using-opencv-part-2/
# By: Mickey Chan @ 2019

# Import required modules
import cv2
import os
import time
import sqlite3

# Connect SQLite3 database
conn = sqlite3.connect("database.db")
db = conn.cursor()
#連接資料庫

# Prepare a directory for storing captured face data
dirName = "./dataset"
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("DataSet Directory Created")
#準備存放照片資料夾

# Ask for the user's name
name = input("What's his/her Name?")

imgCapture = 100 # Number of face image we have to capture
#照片數量
saveFace = False
frameColor = (0,0,255) # Frame color for detected face
#偵測臉部框架顏色
userDir = "User_" # Prefix of face image directory name
#修正後臉部照片資料夾名稱
beginTime = 0

# Connect to video source
#vSource = "rtsp://192.168.1.100:8554/live.sdp" # RTSP URL of IP Cam
vSource = 0 # first USB webcam 
vStream = cv2.VideoCapture(vSource) 
#使用相機

# Setup Classifier for detecting face
faceCascade = cv2.CascadeClassifier("C:\OpenCV\data\haarcascades\haarcascade_frontalface_default.xml")
#載入人臉檢測器（分類器）

# Continuously capture video until collected require amount of face data
count = 1   #起始照片數量
frameRate = 5 # Frequency for capturing face 捕捉頻率
prevTime = 0
while vStream.isOpened():   #相機開機狀態就持續迴圈
    timeElapsed = time.time() - prevTime
    ok, frame = vStream.read() # Read a frame 從攝影機擷取一張影像
    if not ok: break #擷取不到即跳出迴圈
    cv2.putText(frame, "Press 'f' to start face capture", (10, 480-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    if timeElapsed > 1./frameRate: #1/頻率 為秒數。
        prevTime = time.time()
        # Find any face in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert captured frame to grayscale 圖片轉灰階
        faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5) # Detect face(s) inside the frame 使用載入分類器的套件偵測人臉
        # If found, save captured face image
        for (x, y, w, h) in faces: #找到圖片後讀去座標(中心點)
            cv2.rectangle(frame, (x,y), (x+w, y+h), frameColor, 2) # Draw a frame surrounding the face
            #依據中心點畫出正方形
            # Save captured face data
            if saveFace:#抓出灰階照片該方框位置得圖片並存檔
                roiGray = gray[y:y+h, x:x+w]
                fileName = userDir + "/" + f'{count:02}' + ".jpg"
                cv2.imwrite(fileName, roiGray)
                cv2.imshow("face", roiGray)
                count += 1
        
    cv2.imshow('frame', frame) # Show the video frame 顯示影像
    # Press 'f' to begin detect,
    # Press ESC or 'q' to quit
    key = cv2.waitKey(1) & 0xff      #，輸入q結束
    if key == 27 or key == ord('q'):
        break
    elif key == ord('f') and not saveFace: #按下f開始存照片
        saveFace = True
        frameColor = (0, 255, 0)
        beginTime = time.time()
        # Build directory for storing captured faces
        userDir = os.path.join(dirName, userDir + time.strftime("%Y%m%d%H%M%S"))
        if not os.path.exists(userDir):
            os.makedirs(userDir)
    
    # Quit face detection when captured required faces
    if count > imgCapture:   #當counter超過設定數量
        break

# Clean up
vStream.release()   #釋放鏡頭

# Insert a new record
db.execute("INSERT INTO `users` (`name`) VALUES(?)", (name,))
#執行寫入DB
uid = db.lastrowid  #取得最後一個ID位置
print("User ID:" + str(uid))
# Rename temperary directory with USER ID
newUserDir = os.path.join(dirName, str(uid))
os.rename(userDir, newUserDir);
#print("Renamed user dataset directory name to " + newUserDir)
conn.commit()
conn.close()

cv2.destroyAllWindows()
print("DONE")
elapsedTime = round(time.time() - beginTime, 4)
print("Elapsed time: " + str(elapsedTime) + "s")
