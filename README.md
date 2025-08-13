# Self-Driving Car: End-to-End Deep Learning Project
This project is a comprehensive simulation of an autonomous driving system that uses deep learning to perform real-time steering angle prediction, lane segmentation, and object detection from a continuous video feed.

## Overview
The core of this project is a multi-model system designed to replicate the key visual perception tasks of a self-driving car. It processes a dataset of driving images to perform three tasks concurrently:

1. Steering Angle Prediction: A Convolutional Neural Network (CNN), inspired by NVIDIA's DAVE-2 model, predicts the correct steering angle by analyzing the road ahead.

2. Lane Segmentation: A fine-tuned YOLOv8 segmentation model identifies and highlights the drivable lanes on the road.

3. Object Detection: A pre-trained YOLOv8 segmentation model detects and masks various objects, such as other vehicles, to provide environmental awareness.

The final output is a real-time visualization showing the car's camera feed, the combined segmentation overlay, and an animated steering wheel that reflects the model's predictions.

## Tech Stack

This project is built with a robust stack of open-source technologies for machine learning and computer vision.

Language: Python 3.9

Core Deep Learning Frameworks:

TensorFlow 1.x: For building and training the steering angle regression model.

Ultralytics YOLOv8: For training the lane segmentation model and for object detection.

Computer Vision & Data Processing:

OpenCV: For image loading, preprocessing (resizing, cropping), and real-time video display.

NumPy: For efficient numerical operations and data manipulation.

Pandas: For handling the training data from CSV files.

Environment & Package Management:

Conda / venv: For creating isolated and reproducible project environments.

Pip: For installing Python packages.

## Project Workflow

The project is divided into two main phases: Training and Inference.

1. Training Phase
This phase focuses on creating the intelligent models.

Steering Angle Model: The CNN is trained using the train.py script. This script reads a dataset of images and their corresponding steering angles, augments the data (flipping, shadows), and trains the model to minimize the mean squared error between its predictions and the ground truth.

Lane Segmentation Model: The training_lane_deteciton.ipynb notebook uses transfer learning to fine-tune a pre-trained YOLOv8 model on a custom dataset of road lanes, teaching it to accurately segment lane markings.

2. Inference Phase
This is the execution phase where the trained models are put to use.

The main application is launched via the run_fsd_inference command.

It loads all three trained models (steering, lane segmentation, and object detection).

Using Python's concurrent.futures, it processes each frame from the driving dataset in parallel, sending it to the steering model and the segmentation models simultaneously to maximize performance.

The results are combined and displayed in a multi-window GUI, providing a complete simulation of the car's perception system.

## Project Structure

The project is organized into a modular structure to separate concerns like data, model definitions, training scripts, and inference logic.

```
self_driving_car_project/
│
├── data/
│   ├── driving_dataset/      # Images and data.txt for steering model training
│   └── steering_wheel_image.jpg # UI asset
│
├── model_training/
│   ├── train_steering_angle/ # Scripts to train the steering model
│   │   ├── driving_data.py
│   │   ├── model.py
│   │   └── train.py
│   └── training_lane_deteciton.ipynb # Notebook for training lane segmentation
│
├── saved_models/
│   ├── lane_segmentation_model/
│   │   └── best_yolo11_lane_segmentation.pt
│   ├── object_detection_model/
│   │   └── yolo11s-seg.pt
│   └── regression_model/
│       └── model.ckpt.*
│
├── src/
│   ├── __init__.py
│   ├── inference/            # Scripts for running the final application
│   │   ├── run_fsd_inference.py
│   │   ├── run_segmentation_obj_det.py
│   │   └── run_steering_angle_prediciton.py
│   └── models/               # Contains the steering model architecture
│       └── model.py
│
├── .gitignore
├── pyproject.toml
├── requirements.txt          # List of all project dependencies
└── setup.py                  # Makes the project installable with console scripts
```
##  Installation and Setup

Follow these steps to set up the project environment and install all necessary dependencies.

Prerequisites
Python 3.8, 3.9, or 3.10

Conda (or venv) installed

**Step 1: Clone the Repository**
```
git clone [https://github.com/your-username/self_driving_car_project.git](https://github.com/your-username/self_driving_car_project.git)
cd self_driving_car_project
```
**Step 2: Create and Activate a Virtual Environment**
It is highly recommended to create an isolated environment to avoid dependency conflicts.

**Using Conda**
```
# Create the environment
conda create --name fsd_env python=3.9 -y

# Activate the environment
conda activate fsd_env
```
**Step 3: Install Dependencies**
Install all the required packages from the requirements.txt file.
```
pip install -r requirements.txt
```

**Step 4: Install the Project**
Install the project in editable mode. This will set up the console scripts defined in setup.py.

```
pip install -e .
```
## How to Run

With the environment set up and the project installed, you can run the different components.

**Run the Full Simulation (Recommended)**
This command runs the main inference script with all models working concurrently.

```
run_fsd_inference
```
**Run Individual Components**
You can also test each part of the system independently.

To test only the steering angle prediction:
```
run_steering
```

To test only the lane and object segmentation:

```
run_segmentation
```
To stop any simulation, click on one of the display windows and press the 'q' key.

## Training the Models

To retrain the models, you can run the training scripts.

Train the steering model:
```
python model_training/train_steering_angle/train.py
```
Train the lane segmentation model:

Open and run the cells in the model_training/training_lane_deteciton.ipynb Jupyter Notebook.



