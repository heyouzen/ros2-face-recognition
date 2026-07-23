# ROS2 Face Recognition Module

A ROS2-based face detection and recognition module using **YOLOv8-Face** and **ArcFace**.

The project integrates face detection, face feature extraction, identity matching, local face-database management and ROS2 communication for robot vision applications. It is currently developed and tested on Ubuntu 22.04, with the Booster T1 robot as the target deployment platform.

## Features

- Face detection based on YOLOv8-Face
- Face feature extraction based on ArcFace
- Face identity matching based on feature similarity
- Local face-database construction and management
- ROS2 image-topic subscription
- ROS2 recognition-result publication
- Image, video and simulated-camera testing
- OpenCV visualization of recognition results

## System Architecture

```text
Camera / Image / Video
          |
          v
    YOLOv8-Face
    Face Detection
          |
          v
       ArcFace
Feature Extraction
          |
          v
    Face Database
Embedding Similarity Matching
          |
          v
 ROS2 Recognition Result
```

## System Environment

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10
- OpenCV
- Ultralytics
- InsightFace
- ONNX Runtime

## Project Structure

```text
ros2-face-recognition/
├── src/
│   ├── build_face_lib.py
│   ├── detector_yolo.py
│   ├── face_engine.py
│   ├── face_feature.py
│   ├── recognize_face.py
│   ├── ros2_face_node.py
│   ├── run_image.py
│   ├── run_video_recognition.py
│   ├── test_arcface_embedding.py
│   ├── test_camera_pub.py
│   └── video_to_ros_pub.py
├── docs/
│   └── deployment.md
├── .gitignore
└── README.md
```

The following runtime files are stored locally and are not uploaded to GitHub:

```text
models/
face_database/
face_lib.json
test images
test videos
recognition outputs
```

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:heyouzen/ros2-face-recognition.git
cd ros2-face-recognition
```

### 2. Create a Python Environment

```bash
python3 -m venv face_env
source face_env/bin/activate
python -m pip install --upgrade pip
```

### 3. Install Python Dependencies

```bash
pip install ultralytics opencv-python numpy insightface onnxruntime
```

### 4. Install ROS2 Dependencies

```bash
sudo apt update
sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs ros-humble-std-msgs
```

Before running a ROS2 program, load the ROS2 environment:

```bash
source /opt/ros/humble/setup.bash
```

## Model Preparation

This project requires two model components.

### YOLOv8-Face

Place the YOLOv8-Face model at:

```text
models/yolov8s-face-lindevs.pt
```

Example:

```text
models/
└── yolov8s-face-lindevs.pt
```

### InsightFace

The project uses the InsightFace `buffalo_l` model for face-feature extraction.

Recommended directory:

```text
models/insightface/models/buffalo_l/
```

Example:

```text
models/
└── insightface/
    └── models/
        └── buffalo_l/
            ├── det_10g.onnx
            ├── w600k_r50.onnx
            └── other model files
```

Model files are excluded from GitHub and must be prepared locally.

## Face Database Management

The face database is stored locally and is not uploaded to GitHub.

Each person must have an independent directory. Directory names should use lowercase English names or pinyin only.

Example:

```text
face_database/
├── geshirong/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
├── liubo/
│   ├── 01.jpg
│   └── 02.jpg
└── hanshangfeng/
    ├── 01.jpg
    ├── 02.jpg
    └── 03.jpg
```

Do not include personnel numbers or job titles in directory names.

Recommended:

```text
geshirong
liubo
hanshangfeng
```

Avoid:

```text
001_geshirong_yuanshi
002_liubo_xiaozhang
003_hanshangfeng_shuji
```

### Face Image Recommendations

For each person, prepare approximately 5 to 10 images.

Recommended conditions:

- Clear and visible face
- Frontal and slightly angled views
- Moderate changes in lighting and expression
- One main person in each image
- No severe blur or heavy occlusion
- No duplicated copies of the same image

### Build the Face Feature Library

After adding or changing face images, run:

```bash
python -m src.build_face_lib
```

The program reads images from:

```text
face_database/
```

and generates:

```text
face_lib.json
```

The generated file stores the face embeddings used for identity matching.

After rebuilding `face_lib.json`, restart the recognition program.

## Running the Recognition Module

### Test the Face Engine

```bash
python -m src.face_engine
```

### Image Recognition

```bash
python -m src.run_image
```

### Video Recognition

```bash
python -m src.run_video_recognition
```

## ROS2 Recognition Test

The complete ROS2 test requires three terminals.

### Terminal 1: Start the Recognition Node

```bash
source /opt/ros/humble/setup.bash
source face_env/bin/activate
python -m src.ros2_face_node
```

The recognition node subscribes to:

```text
/camera/image_raw
```

Message type:

```text
sensor_msgs/msg/Image
```

### Terminal 2: Start the Test Image Publisher

```bash
source /opt/ros/humble/setup.bash
source face_env/bin/activate
python -m src.test_camera_pub
```

The test publisher sends a local image to:

```text
/camera/image_raw
```

### Terminal 3: View Recognition Results

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /face/recognition_result
```

## ROS2 Interface

### Input Topic

```text
/camera/image_raw
```

Input message type:

```text
sensor_msgs/msg/Image
```

### Output Topic

```text
/face/recognition_result
```

Output message type:

```text
std_msgs/msg/String
```

The current recognition result contains:

- `bbox`: face bounding-box coordinates
- `det_confidence`: face-detection confidence
- `name`: recognized person name
- `similarity`: face-feature similarity score

Example:

```json
{
  "bbox": [100, 39, 255, 243],
  "det_confidence": 0.91,
  "name": "geshirong",
  "similarity": 0.80
}
```

## Visualization

When OpenCV visualization is enabled, the output window displays:

- Face bounding box
- Recognized person name
- Similarity score

Example:

```text
geshirong 0.80
```

## Adding a New Person

1. Create a new directory:

```bash
mkdir -p face_database/new_person
```

2. Add face images:

```text
face_database/new_person/
├── 01.jpg
├── 02.jpg
├── 03.jpg
├── 04.jpg
└── 05.jpg
```

3. Rebuild the feature library:

```bash
python -m src.build_face_lib
```

4. Restart the recognition node.

5. Test the new identity through image, video or ROS2 camera input.

## GitHub Data Policy

GitHub stores:

- Source code
- README documentation
- Deployment documentation
- Git configuration files

GitHub does not store:

- Face images
- Face embeddings
- Model weights
- Private test images
- Private test videos
- Recognition outputs

These files are excluded through `.gitignore`.

## Deployment Guide

Detailed environment configuration, model placement, database management and ROS2 deployment instructions are available in:

[Deployment Guide](docs/deployment.md)

## Current Project Status

Completed:

- YOLOv8-Face detection
- ArcFace feature extraction
- Face-library matching
- Local database generation
- Image and video testing
- ROS2 image subscription
- ROS2 result publication
- Simulated-camera testing
- OpenCV visualization
- GitHub repository management
- Deployment documentation

Remaining work:

- Expand the face database
- Test additional known identities
- Test unknown identities
- Evaluate the recognition threshold
- Complete robot-camera integration when the robot platform is available

## Notes

- Some scripts may still contain local absolute paths. Update these paths before deploying the project on another computer.
- Model files and face data must be prepared locally.
- The shared project database should be maintained by the team on the designated virtual machine.
- Robot-camera integration is reserved for the later deployment stage.
