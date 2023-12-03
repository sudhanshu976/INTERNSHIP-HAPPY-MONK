from ultralytics import YOLO
import cv2
import numpy as np
import math
import cvzone

cap = cv2.VideoCapture("video1.mp4")
cap.set(3, 1280)
cap.set(4, 720)

classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bs', 'train', 'trck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'mbrella', 'handbag', 'tie', 'sitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat'
    , 'baseball glove',
    'skateboard', 'srfboard', 'tennis racket', 'bottle', 'wine glass', 'cp', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'dont', 'cake', 'chair', 'coch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mose', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrsh'
]

model = YOLO('yolov8n.pt')

while True:
    success, img = cap.read()
    results = model(img , stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            w , h = x2-x1 , y2-y1
            bbox = int(x1),int(y1),int(w),int(h)

            #Confidence
            conf = math.ceil((box.conf[0]*100))/100

            #CLASS
            cls = int(box.cls[0])
            currentclass = classNames[cls]
            if currentclass == "person" and conf>0.3:
                currentArray = np.array([x1,y1,x2,y2,conf])
                #cvzone.cornerRect(img,bbox , l=10 , t=3)
                cv2.rectangle(img , (bbox) , (0,255,0) , 5)
                cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()