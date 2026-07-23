ROS2 Face Recognition Module
A ROS2-based face detection and recognition module using YOLOv8-Face and ArcFace.
This project integrates face detection, face feature extraction, face database management and ROS2 communication for robot vision applications.
Features
Face detection based on YOLOv8-Face
Face feature extraction based on ArcFace
Face identity recognition using feature similarity matching
ROS2 node publishing recognition results
Support image, video and camera input
System Environment
Ubuntu 22.04
ROS2 Humble
Python 3.10
Framework
Camera/Image
      |
      v
YOLOv8-Face
(Face Detection)
      |
      v
ArcFace
(Face Feature Extraction)
      |
      v
Face Database
(Face Embedding Matching)
      |
      v
ROS2 Topic Output
Project Structure
ros2-face-recognition/

├── src/
│   ├── detector_yolo.py
│   ├── face_feature.py
│   ├── face_engine.py
│   ├── recognize_face.py
│   ├── ros2_face_node.py
│   └── run_xxx.py
│
├── models/
│   └── YOLOv8-face model files
│
├── face_database/
│   └── personal face images
│
├── face_lib.json
│
└── README.md
Installation
Clone the repository:
git clone git@github.com:heyouzen/ros2-face-recognition.git
cd ros2-face-recognition
Install dependencies:
pip install -r requirements.txt
Model Preparation
Download the YOLOv8-Face pretrained model.
Place the model file into:
models/
Example:
models/
└── yolov8-face.pt
Face Database Management
The face database is stored locally and is not uploaded to GitHub.
Database structure:
face_database/

├── person1/
│   ├── 001.jpg
│   ├── 002.jpg
│
├── person2/
│   ├── 001.jpg
To add a new person:
Create a new folder with the person's name.

Add several face images.

Generate the face feature library:

python src/build_face_lib.py
The generated feature library:
face_lib.json
will be used for face recognition.
Running
Image Recognition
python src/run_image.py
Video Recognition
python src/run_video_recognition.py
ROS2 Node
python src/ros2_face_node.py
ROS2 Output
The recognition result contains:
Person name
Similarity score
Bounding box information
Notes
Face images are stored locally and are not uploaded to GitHub.
YOLOv8-Face model needs to be downloaded separately.
Each deployment machine should prepare its own face database.
