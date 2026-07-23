# ROS2 Face Recognition Deployment Guide

## 1. Project Overview

This project provides a ROS2-based face detection and recognition module for robot vision applications.

The system includes:

- Face detection based on YOLOv8-Face
- Face feature extraction based on ArcFace
- Face identity matching based on cosine similarity
- ROS2 image subscription and recognition-result publication
- OpenCV visualization of recognition results

The current version has been tested in an Ubuntu 22.04 virtual machine. The target deployment platform is the Booster T1 robot.

## 2. Current System Environment

### Hardware

- Development environment: VMware virtual machine
- Target robot: Booster T1
- Target computing platform: NVIDIA AGX Orin V2.1

### Software

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10
- OpenCV
- Ultralytics
- InsightFace
- ONNX Runtime

## 3. Clone the Repository

```bash
git clone git@github.com:heyouzen/ros2-face-recognition.git
cd ros2-face-recognition
```

## 4. Create the Python Environment

Create a Python virtual environment:

```bash
python3 -m venv ~/face_env
```

Activate the environment:

```bash
source ~/face_env/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the main Python dependencies:

```bash
pip install ultralytics opencv-python numpy insightface onnxruntime
```

Install the required ROS2 packages:

```bash
sudo apt update
sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs ros-humble-std-msgs
```

## 5. Project Path Configuration

The current code was developed under:

```text
/home/chen/yolo
```

Some Python files may contain absolute paths beginning with:

```text
/home/chen/yolo
```

When deploying the project on another computer, check and update the paths in the following files:

- `src/face_engine.py`
- `src/face_feature.py`
- `src/recognize_face.py`
- `src/test_camera_pub.py`
- Other test scripts containing local image or model paths

The paths should point to the actual project directory on the deployment computer.

## 6. Model Preparation

The project requires two model components:

- YOLOv8-Face model for face detection
- InsightFace `buffalo_l` model for face feature extraction

### YOLOv8-Face Model

Place the YOLOv8-Face weight file at:

```text
models/yolov8s-face-lindevs.pt
```

Example model directory:

```text
models/
└── yolov8s-face-lindevs.pt
```

### InsightFace Model

The InsightFace model is stored locally and is not uploaded to GitHub.

The current model directory is:

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

InsightFace may download the model automatically during the first run. If automatic downloading fails, copy the complete `buffalo_l` directory into the path shown above.

## 7. Team Face Database Location

The face-image database is maintained locally on the shared virtual machine and is not uploaded to GitHub.

The current database directory is:

```text
/home/chen/yolo/face_database
```

The generated face-feature library is:

```text
/home/chen/yolo/face_lib.json
```

Recommended directory structure:

```text
face_database/
├── geshirong/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
├── person_name_2/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
└── person_name_3/
    ├── 01.jpg
    ├── 02.jpg
    └── 03.jpg
```

Use lowercase English names or pinyin for person-folder names.

Do not add numeric identity prefixes such as:

```text
001_geshirong
002_person_name
```

Use:

```text
geshirong
person_name
```

## 8. Face Image Requirements

For each person, prepare approximately 5 to 10 images.

Recommended image conditions:

- The face should be clear and visible
- The image should contain one main person
- Use frontal and slightly angled faces
- Include moderate changes in lighting and expression
- Avoid severe blur
- Avoid heavy occlusion
- Avoid very small faces
- Avoid duplicated copies of exactly the same image

## 9. Build or Update the Face Library

Activate the Python environment:

```bash
cd ~/yolo
source /home/chen/face_env/bin/activate
```

Run the database-building script:

```bash
python -m src.build_face_lib
```

If module execution is not supported by the current script, run:

```bash
python src/build_face_lib.py
```

The script reads images from:

```text
face_database/
```

and generates or updates:

```text
face_lib.json
```

After adding a new person, rebuild `face_lib.json` and restart the recognition program.

Before rebuilding the database, keep the original images for all existing people inside `face_database/`. This prevents existing identities from being removed when the feature library is regenerated.

## 10. Test the Recognition Engine

Activate the environment:

```bash
cd ~/yolo
source /home/chen/face_env/bin/activate
```

Run the face engine test:

```bash
python -m src.face_engine
```

A successful result should contain information similar to:

```text
bbox
det_confidence
name
similarity
```

Example:

```text
name: geshirong
similarity: 0.80
```

## 11. Start the ROS2 Recognition Node

Open the first terminal:

```bash
cd ~/yolo
source /opt/ros/humble/setup.bash
source /home/chen/face_env/bin/activate
python -m src.ros2_face_node
```

The node subscribes to:

```text
/camera/image_raw
```

The input message type is:

```text
sensor_msgs/msg/Image
```

The node publishes recognition results to:

```text
/face/recognition_result
```

The output message type is:

```text
std_msgs/msg/String
```

## 12. Start the Virtual Camera Publisher

When the robot camera is unavailable, use the local test publisher.

Open the second terminal:

```bash
cd ~/yolo
source /opt/ros/humble/setup.bash
source /home/chen/face_env/bin/activate
python -m src.test_camera_pub
```

The test publisher sends a local image to:

```text
/camera/image_raw
```

## 13. View the ROS2 Recognition Results

Open the third terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /face/recognition_result
```

Check the active topics:

```bash
ros2 topic list
```

The expected topics include:

```text
/camera/image_raw
/face/recognition_result
```

Check the recognition topic information:

```bash
ros2 topic info /face/recognition_result
```

## 14. Recognition Output Format

The current recognition result contains:

- `bbox`: face bounding-box coordinates
- `det_confidence`: YOLOv8-Face detection confidence
- `name`: recognized person name
- `similarity`: ArcFace feature similarity

Example:

```json
{
  "bbox": [100, 39, 255, 243],
  "det_confidence": 0.91,
  "name": "geshirong",
  "similarity": 0.80
}
```

The current version does not output a separate `status` field.

## 15. Visualization

When visualization is enabled, the OpenCV window displays:

- The detected face bounding box
- The recognized person name
- The similarity score

Example:

```text
geshirong 0.80
```

Known identities are displayed with a green bounding box.

## 16. Add a New Person

Follow these steps to add a new person.

### Step 1: Create the Person Directory

Example:

```bash
mkdir -p ~/yolo/face_database/new_person
```

### Step 2: Add Face Images

Place the images inside:

```text
face_database/new_person/
```

Example:

```text
face_database/new_person/
├── 01.jpg
├── 02.jpg
├── 03.jpg
├── 04.jpg
└── 05.jpg
```

### Step 3: Rebuild the Face Library

```bash
cd ~/yolo
source /home/chen/face_env/bin/activate
python -m src.build_face_lib
```

### Step 4: Restart the Recognition Node

Stop the old node with:

```text
Ctrl + C
```

Then restart it:

```bash
source /opt/ros/humble/setup.bash
source /home/chen/face_env/bin/activate
python -m src.ros2_face_node
```

### Step 5: Test the New Identity

Publish an image or video containing the new person and verify that the correct name appears in:

```text
/face/recognition_result
```

## 17. Files Not Uploaded to GitHub

The following files and directories are maintained locally:

```text
models/
face_database/
face_lib.json
face_lib_backup*.json
outputs/
logs/
test_data/videos/
*.jpg
*.png
*.pt
*.onnx
```

These files are excluded by `.gitignore`.

## 18. Team Collaboration Workflow

The GitHub repository stores:

- Source code
- README documentation
- Deployment documentation
- Git configuration files

The shared virtual machine stores:

- Face images
- Face feature library
- YOLOv8-Face weights
- InsightFace model files
- Test images and videos
- Recognition outputs

When code is updated:

```bash
git pull
```

When local code changes need to be uploaded:

```bash
git add .
git commit -m "Describe the update"
git push
```

Do not upload face images, face embeddings, model weights or private test data.

## 19. Current Completion Status

Completed:

- YOLOv8-Face detection
- ArcFace feature extraction
- Face-library matching
- ROS2 image subscription
- ROS2 result publication
- Local image simulation
- OpenCV visualization
- GitHub source-code management
- README documentation
- Deployment documentation

Remaining work:

- Expand the face database
- Test multiple known identities
- Test unknown identities
- Evaluate the similarity threshold
- Perform final robot-camera integration when the robot interface is available
