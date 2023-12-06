# Project Documentation

## Task 1 - YOLOv8 Object Detection for Person

### Task 1A
**Objective:** Implement YOLOv8 object detection model to identify and locate specifically the person class in images.

**Implementation:**
- **Library Used:** Ultralytics (YOLOv8), OpenCV, Numpy, CVZONE.
- **Files in Repository:**
  - `task.py`: Python script for YOLOv8 person detection.
  - `yolov8.pt`: YOLOv8 pre-trained weights.
- **Execution:**
  - Source images/videos are in the "source video" folder.
  - Results are stored in the "results video" folder.
  - COCO dataset class names are used to filter for the person class.

### Task 1B
**Objective:** Fine-tune a pre-trained MobileNetV2 model on a custom dataset for plant classification.

**Implementation:**
- **Library Used:** TensorFlow, OpenCV, Numpy, Matplotlib, MobileNetV2.
- **Files in Repository:**
  - `dataset.txt`: Link to the PlantVillage Dataset for tomatoes.
  - `notebook.ipynb`: Jupyter notebook with the implementation.
  - `model.h5`: Trained MobileNetV2 model.
- **Execution:**
  - Data augmentation applied using TensorFlow layers.
  - MobileNetV2 selected for its suitability for edge devices.
  - Model trained for 15 epochs, achieving 87% validation accuracy.

## Task 2 - YOLOv5 Object Detection for Face and Emotion

### Task 2A
**Objective:** Train a YOLOv5 object detection model for face detection.

**Implementation:**
- **Library Used:** Ultralytics (YOLOv5), OpenCV, Numpy, PyTorch.
- **Files in Repository:**
  - `MASK DETECTOR.ipynb`: Jupyter notebook for YOLOv5 face detection.
  - `yolov5`: YOLOv5 GitHub repository.
- **Execution:**
  - Custom dataset.yaml file created for the provided face detection dataset.
  - YOLO nano model used, trained for 16 epochs with 74% mAP.

### Task 2B
**Objective:** Train an image classification model to classify emotions in face crops.

**Implementation:**
- **Library Used:** Ultralytics (YOLOv5), OpenCV, Numpy, PyTorch.
- **Files in Repository:**
  - `EMOTION DETECTOR.ipynb`: Jupyter notebook for emotion classification.
- **Execution:**
  - Data collection using webcam for classes Happy, Sad, Neutral.
  - Annotated images using LabelImg in YOLO format.
  - Custom YOLO model trained for 100 epochs, achieving good results for Happy and Neutral classes.

## Task 4 - Vehicle Counting and Speed Estimation

**Objective:** Use YOLOv8 object detection to identify vehicles, count them, and estimate their speed in a video.

**Implementation:**
- **Library Used:** Ultralytics (YOLOv8), OpenCV, Numpy, CVZONE, TIME, SORT (GitHub repo).
- **Files in Repository:**
  - `car_counter.py`: Vehicle counting and speed estimation script.
  - `co-ordinates.py`: Co-ordinates finder script.
  - `requirements.txt`: Dependencies file.
  - `sort.py`: SORT algorithm for object tracking.
  - `yolov8.pt`: YOLOv8 pre-trained weights.
- **Execution:**
  - Limited YOLO classes to detect only vehicles.
  - Applied a mask to focus detection on a specific region.
  - Used SORT for real-time vehicle tracking with unique IDs.
  - Implemented a method to estimate vehicle speed using pixel-to-meter conversion.

## General Notes

- All code, source images/videos, and results are available in the GitHub repository.
- YOLOv5 was used in Task 2 instead of YOLOv8 due to technical issues.
- Models are not trained for longer due to time constraints, affecting accuracy.
- Only edge device-preferable models like YOLO nano/small or MobileNet were used.
- Video proofs for Task 1B were limited due to time constraints.
- The most challenging part of Task 2B was data collection and annotation.
- Task 4 involved finding an approach to accurately estimate vehicle speed, with pixel-to-meter conversion used in the implementation.
