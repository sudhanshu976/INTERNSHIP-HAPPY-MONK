# NECESSARY IMPORTS
from ultralytics import YOLO
import cv2
import cvzone
from sort import *
import math
import time
import numpy as np

# VIDEO CAPTURE
cap = cv2.VideoCapture("footage.mp4")
cap.set(3, 1280)
cap.set(4, 720)

# SPEED ESTIMATION FUNCTIONS
prev_centroids = {}
pixels_to_meters = 0.1 
prev_time = time.time()
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# COCO DATASET CLASSES
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

# MASK
mask = cv2.imread("mask.png")

# TRACKER ID
tracker = Sort(max_age=20 , min_hits = 3 , iou_threshold=0.3)

# YOLO MODEL
model = YOLO('yolov8n.pt')

#TEXT FORMATTING 
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4      
font_color = (255, 255, 0)
font_thickness = 1

# CORDINATES/LIMITS OF THE LINE
limits1  = [524,346,1496,298]


# COUNTER
tcount = []

# MAIN CODE
while True:
    success, img = cap.read()

    # Resize the mask and fits it
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imgRegion = cv2.bitwise_and(img,mask)

    # Detecting
    results = model(imgRegion , stream=True)

    # Detections for tracker
    detections = np.empty((0,5))

    # Looping to get coordinates of the bounding boxes
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
            if currentclass == "car" or currentclass=="truck" or currentclass == "bus" or currentclass == "motorbike" and conf>0.3:
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))


    # TRACKING DETECTIONS AND GIVING UNIQUE ID TO EACH CAR
    resultsTracker = tracker.update(detections)
    current_time = time.time()
    elapsed_time = current_time - prev_time
    prev_time = current_time

    # Loop through tracked results and calculate speed
    for results in resultsTracker:
        x1, y1, x2, y2, id = results
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2

        # Check if the ID is in the dictionary
        if id in prev_centroids:
            # Calculate distance between current and previous centroid (in meters)
            distance_pixels = calculate_distance(cx, cy, prev_centroids[id][0], prev_centroids[id][1])
            distance_meters = distance_pixels * pixels_to_meters

            # Calculate speed (in meters per second)
            speed_meters_per_second = distance_meters / elapsed_time

            # Convert speed to kilometers per hour
            speed_kmh = speed_meters_per_second * 3.6

            # Display the speed on the screen
            cv2.putText(img, f"ID: {id}, Speed: {speed_kmh:.2f} km/h", (int(x1), int(y1) - 40),
                        font, font_scale, font_color, font_thickness)

        # Update the dictionary with the current centroid
        prev_centroids[id] = (cx, cy)
    cv2.line(img , (limits1[0] , limits1[1]) , (limits1[2],limits1[3]) , (0,0,255) , 5)
    

    # COUNTING CAR CODE
    for results in resultsTracker:
        x1,y1,x2,y2,id = results
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        w , h = x2-x1 , y2-y1

        cvzone.cornerRect(img,(x1,y1,w,h) , l=10 , t=3)
        cv2.putText(img, f"{id}  {conf}", (int(x1), max(0, int(y1) - 20)), font, font_scale, font_color, font_thickness)
        cx , cy = x1+w//2 , y1+h//2
        cv2.circle(img , (cx,cy) , 5 , (255,0,0) , cv2.FILLED)

        if limits1[0]< cx < limits1[2] and limits1[1]-10 < cy < limits1[1]+10:
            if tcount.count(id)==0:
                tcount.append(id)

                
    cvzone.putTextRect(img, f"COUNT : {len(tcount)}", (50,50))

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()