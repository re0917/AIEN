from django.shortcuts import render,redirect,HttpResponse
from django.urls import reverse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from . import yolo_tiny
import PIL.Image
import numpy as np
import cv2
import time
import call
from django.views.decorators.csrf import csrf_exempt # import for csrf

# Create your views here.
finalresult={}
def button(request):
    labelsPath = r"/home/test/prejectweb/prejectweb/item/yolo-coco/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    weightsPath = r"/home/test/prejectweb/prejectweb/item/yolo-coco/yolov3-tiny.weights"
    configPath = r"/home/test/prejectweb/prejectweb/item/yolo-coco/yolov3-tiny.cfg"

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    image = cv2.imread(r"/home/test/prejectweb/prejectweb/static/images/1.jpg")
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]


            if confidence > 0.1:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
        0.5)
    foritem = []
    foritlab = ""
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
        
            # print(foritem)
            print(LABELS[classIDs[i]])
            foritem.append(LABELS[classIDs[i]])
            print(foritem)

            global finalresult    
            finalresult[LABELS[classIDs[i]]]=confidences[i]
            print(finalresult)

    if len(idxs)>0:
        for j in foritem:
            foritlab += j + ' '
        foritlab+='忘在家裡了！'
        call.lineNotifyMessage(foritlab)
    else:
        c='隨身物品已帶齊'
        call.lineNotifyMessage(c)


    #cv2.imshow("Image", image)
    cv2.waitKey(0)
    time.sleep(1000)
    return render(request, 'profile.html')

def up(request):
    return render(request, 'profile.html')

@csrf_exempt
def upload(request):   
    reqfile = request.FILES['file'] 
    image = PIL.Image.open(reqfile) 
    image.save(r"/home/test/prejectweb/prejectweb/static/images/1.jpg","jpeg") 
    return render(request, 'profile.html')   

def send_data(request):
    result=finalresult
    return JsonResponse({"data":result})