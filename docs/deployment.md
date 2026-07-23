# ROS2 Face Recognition Deployment Guide

## 1. Project Overview

This project provides a ROS2-based face detection and recognition module for the Booster T1 robot.

The system contains:

- Face detection: YOLOv8-Face
- Face feature extraction: ArcFace
- Face recognition: feature similarity matching
- Robot communication: ROS2 node


---

# 2. Environment

## Hardware

- Booster T1 robot platform
- NVIDIA AGX Orin V2.1


## Software

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10+


---

# 3. Installation

## Clone Repository

```bash
git clone git@github.com:heyouzen/ros2-face-recognition.git

cd ros2-face-recognition
Python Environment
Create environment:
conda create -n face_env python=3.10

conda activate face_env
Install required packages:
pip install -r requirements.txt
4. Model Preparation
The project uses YOLOv8-Face for face detection.
Download the YOLOv8-Face weight file and place it in the model directory.
Example:
models/

└── yolov8-face.pt
The model file is not included in GitHub because of its size.
5. Face Database Management
Database Structure
Face images should be stored locally.
Example:
face_database/

├── person_1/

│   ├── 001.jpg

│   ├── 002.jpg


├── person_2/

│   ├── 001.jpg

│   └── 002.jpg
Recommended:
5-10 images per person
Clear face images
Different angles and lighting conditions
Generate Face Feature Library
After adding new people, run:
python src/build_face_lib.py
The program will generate:
face_lib.json
The file stores face feature embeddings for recognition.
6. Running Recognition
Image Recognition
python src/run_image.py
Video Recognition
python src/run_video_recognition.py
ROS2 Recognition Node
Run:
python src/ros2_face_node.py
7. Adding New People
To add a new person:
Step 1
Add face images:
face_database/new_person/
Step 2
Regenerate face feature library:
python src/build_face_lib.py
Step 3
Restart recognition node.
The new person can then be recognized.
8. ROS2 Interface
The recognition node publishes face recognition results through ROS2 topics.
Output information includes:
person_id
name
confidence
track_id
is_known
9. GitHub Management
The following files are not uploaded to GitHub:
face_database/

face_lib.json

models/*.pt

test_data/
These files should be maintained locally.
10. Notes
Before running the system, make sure:
YOLOv8-Face model weights are available.

Face database has been generated.

Python environment and ROS2 environment are correctly configured.
