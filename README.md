# Computer Vision - Person Detection in Videos using AWS Rekognition

This project implements a simple computer vision system to detect people in videos using AWS Rekognition. The system processes a video, analyzes selected frames, and generates annotations along with images with and without bounding boxes for each analyzed frame.

---

## Description

The script:

- Loads a local video file (`subway.mp4`).
- Processes 1 out of every 3 frames to optimize cost and analysis time.
- Sends each selected frame to AWS Rekognition for object detection.
- Filters detections to only include the class "Person".
- Saves detection annotations in `.txt` files (format: center_x, center_y, width, height).
- Saves original frames and the same frames with bounding boxes drawn in separate folders.

---

## Requirements

- Python 3.x
- Boto3
- OpenCV (cv2)
- Valid AWS credentials with Rekognition permissions

Install required libraries via pip:

```bash
pip install boto3 opencv-python
