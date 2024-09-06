
# **Person Detection and Tracking in Group Therapy for Autism Spectrum Disorder**

This project uses YOLOv8 for real-time person detection and Deep SORT for tracking individuals in a video of group therapy sessions for children with Autism Spectrum Disorder (ASD). The main objective is to identify and track children and therapists, enabling a deeper analysis of interactions and behaviors.

## **Requirements**

To run this project, you need the following Python libraries:

- `ultralytics` (for YOLOv8)
- `opencv-python-headless`
- `filterpy`
- `numpy`
- `matplotlib`
- `deep_sort_realtime`

You can install these dependencies using the following commands:

```bash
pip install ultralytics  # for YOLOv8 (includes YOLOv5)
pip install opencv-python-headless
pip install filterpy
pip install numpy
pip install matplotlib
pip install deep_sort_realtime
```

## **Usage**

### **1. Model Initialization**
The YOLOv8 model is loaded for object detection:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8 model
```

### **2. Video Processing**
Load your video file and initialize the video writer:
```python
import cv2

video_path = 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
```

### **3. Detection and Tracking**
Process each frame to detect and track individuals:
```python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    tracker = DeepSort(max_age=30)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append([x1, y1, x2-x1, y2-y1, box.conf, box.cls])

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if track.is_confirmed():
            x1, y1, x2, y2 = track.to_tlbr()
            track_id = track.track_id
            label = f"{result.names[box.cls]} ID: {track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)
```

### **4. Saving the Output**
Release resources and save the annotated video:
```python
cap.release()
out.release()
cv2.destroyAllWindows()
```

The output video will be saved as `output_video.mp4` in the current working directory, containing bounding boxes around detected individuals along with their labels and tracking IDs.

## **Project Structure**

- `main.py`: The main script for running the detection and tracking.
- `README.md`: Documentation of the project (this file).
- `requirements.txt`: List of dependencies.

## **Conclusion**

This project demonstrates the integration of YOLOv8 and Deep SORT for detecting and tracking individuals in group therapy sessions for ASD. It provides a tool for analyzing behaviors and interactions in these settings.
