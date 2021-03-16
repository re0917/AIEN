from django.shortcuts import render,redirect,HttpResponse
from django.urls import reverse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import sqlite3
import cv2
import time
# Create your views here.
def button(request):

   # 今天先不探討什麼是 render，先記得它會去撈 test.html
   return render(request, 'button.html')
finalresult={}

def run(request):
    conn = sqlite3.connect('database.db')
    db = conn.cursor()

    # Assign the training data file
    fname = "recognizer/trainingData.yml"        #讀取訓練好的檔案
    if not os.path.isfile(fname):
        print("Please train the data first")    #讀不到訓練檔案後的訊息
        exit(0)

    # Setup GPIO for door lock
    #relayPin = 26
    #GPIO.setmode(GPIO.BCM)
    #GPIO.setup(relayPin, GPIO.OUT)
    #GPIO.output(relayPin, 0)

    lastDetectedAt = 0
    detectInterval = 5 # 1/n second, for reducing overhead
    lastUnlockedAt = 0
    unlockDuration = 5 # n second

    # Font used for display
    font = cv2.FONT_HERSHEY_SIMPLEX   #選擇字體

    # Connect to video source
    #vSource = "rtsp://192.168.1.100:8554/live.sdp" # RTSP URL of IP Cam
    vSource = 0 # first USB webcam
    vStream = cv2.VideoCapture(vSource)    #開啟鏡頭

    # Setup Classifier for detecting face
    faceCascade = cv2.CascadeClassifier("/home/test/prejectweb/prejectweb/face/haarcascade_frontalface_default.xml") #載入分類器
    
    # Setup LBPH recognizer for face recognition
    recognizer = cv2.face.LBPHFaceRecognizer_create() # or LBPHFaceRecognizer_create() 載入人臉識別器
    # Load training data
    recognizer.read(fname) # change to read() for LBPHFaceRecognizer_create() 載入訓練檔案
    count=0

    while vStream.isOpened():   #鏡頭開啟時持續執行迴圈
        # Lock the door again when timeout
        #if time.time() - lastUnlockedAt > unlockDuration:
        #    GPIO.output(relayPin, 0)
        
        ok, frame = vStream.read() # Read frame
        if not ok: break
        
        timeElapsed = time.time() - lastDetectedAt
        if timeElapsed > 1./detectInterval:
            lastDetectedAt = time.time()
            
            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert captured frame to grayscale 轉灰階
            faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5) # Detect face(s) inside the frame
            #Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
            #偵測臉部並且畫出他的正方形四角座標
            for (x, y, w, h) in faces:
                # Try to recognize the face using recognizer
                roiGray = gray[y:y+h, x:x+w]  #截取測試影片人臉位置
                id_, conf = recognizer.predict(roiGray)  #與訓練資料的人臉做比較預測
                print(id_, conf)
                
                # If recognized face has enough confident (<= 70),
                # retrieve the user name from database,
                # draw a rectangle around the face,
                # print the name of the user and
                # unlock the door for 5 secords
                if conf <= 60:  #數字越小代表信心指數越高
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # retrieve user name from database
                    db.execute("SELECT `name` FROM `users` WHERE `id` = (?);", (id_,))
                    #進入資料庫搜尋id
                    result = db.fetchall()
                    #返回多个元组，即返回多条记录(rows),如果没有结果,则返回 ()
                    name = result[0][0]
                    
                    # You may do anything below for detected user, e.g. unlock the door
                    #GPIO.output(relayPin, 1) # Unlock
                    lastUnlockedAt = time.time()
                    print("[Welcome] " + str(id_) + ":" + name + " (" + str(conf) + ")")
                    cv2.putText(frame, name, (x+2,y+h-5), font, 1, (150,255,0), 2)
                    #輸出偵測結果到畫面上
                    count=0

                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    #GPIO.output(relayPin, 0) # Lock the door if not enough confident
                    #print("[Lock] " + name + " " + str(conf))
                    cv2.putText(frame, 'No Match', (x+2,y+h-5), font, 1, (0,0,255), 2)
                    #輸出偵測結果到畫面上
                    count+=1
                    name="未知人士"
                    conf=0
                global finalresult
                
                finalresult[name]=conf
        
        cv2.namedWindow('Face Recognizer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Recognizer',800,700)
        cv2.moveWindow("Face Recognizer", 800, 250)
        cv2.imshow("Face Recognizer", frame)
        
        # Press ESC or 'q' to quit the program
        key = cv2.waitKey(1) & 0xff
        if key == 27 or key == ord('q'):
            break

    # Clean up
    vStream.release()
    conn.close()
    cv2.destroyAllWindows()
    #GPIO.cleanup()
    print("END")
    return render(request, 'button.html')
 #================================================================

 
def send_data(request):
    result=finalresult
    return JsonResponse({"data":result})