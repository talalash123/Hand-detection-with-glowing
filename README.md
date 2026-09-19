# Hand Detection with Glowing & Fire Particle Visual Effects

## Overview

This project provides a real-time computer vision pipeline for hand tracking and dynamic visual overlay rendering using MediaPipe and OpenCV. It detects human hand joint landmarks via a live webcam feed and applies custom glowing bone overlays along with physical fire particle simulations that float upward in real time.

The application leverages classical computer vision techniques combined with machine learning landmark estimation to achieve smooth frame rates, high spatial precision, and interactive visual feedback.

---

## Key Features

- **Real-Time Hand Landmark Tracking:** Tracks full 21-point hand skeleton landmarks with sub-pixel precision across multi-hand streams.
- **Dynamic Glow Overlay:** Renders soft Gaussian blur glow masks around detected hand connections.
- **Fire Particle Simulation System:** Spawns upward-rising flame particles at finger joints with randomized velocity, radius decay, and color-gradient transitions (Yellow to Flame Orange to Dark Red).
- **Dual Mask Blending Pipeline:** Combines raw camera frames with layered line and glow masks using weighted alpha blending (`cv2.addWeighted`).
- **Low Latency Pipeline:** Runs efficiently on standard CPU configurations without requiring heavy GPU rendering engines.

---

## Technical Architecture

The processing pipeline follows a modular execution flow:

1. **Frame Capture & Preprocessing:**
   - Reads camera stream from OpenCV (`cv2.VideoCapture`).
   - Flips frame horizontally for natural mirror interaction.
   - Converts color space from BGR to RGB for MediaPipe integration.

2. **Landmark Extraction:**
   - Processes input frame through `mediapipe.python.solutions.hands`.
   - Converts normalized landmark coordinates (x, y) to pixel space dimensions (W, H).

3. **Mask Generation & Particle Engine:**
   - Constructs base skeleton connections across thumb, index, middle, ring, and pinky fingers.
   - Generates fire particles at key joint positions using randomized kinematic variables (v_x, v_y, decay).
   - Updates active particle positions and removes expired particles from memory per iteration.

4. **Image Blending & Post-Processing:**
   - Applies 2D Gaussian Kernel filtering to particle and glow masks.
   - Merges processed visual layers back onto original frame buffer.

---

## Project Directory Structure

```text
.
├── main.py              # Main execution script (Hand tracking & particle simulation)
├── best.pt              # Trained YOLO pose/detection model weights
├── .gitignore           # Git exclusion configurations
└── README.md            # Project documentation
Installation & Environment Setup
Prerequisites
Ensure Python 3.8+ is installed on your system.

1. Clone the Repository
Bash
git clone https://github.com/talalash123/Hand-detection-with-glowing.git
cd Hand-detection-with-glowing
2. Install Required Dependencies
Install the exact package requirements to avoid MediaPipe API compatibility issues:

Bash
pip install opencv-python numpy mediapipe==0.10.14
Usage Instructions
To launch the real-time hand tracker and fire particle visualizer, execute:

Bash
python main.py
Controls
Press q: Terminate execution and close video display windows.

Performance Evaluation & Metrics
The baseline model evaluation on custom hand detection dataset yielded the following validation metrics:

Bounding Box Detection Metrics
Precision: 97.65%

Recall: 98.06%

mAP @ 0.50: 99.10%

mAP @ 0.50:0.95: 88.04%

Pose Keypoint Metrics
Precision: 86.56%

Recall: 83.17%

mAP @ 0.50: 81.30%

mAP @ 0.50:0.95: 65.28%
