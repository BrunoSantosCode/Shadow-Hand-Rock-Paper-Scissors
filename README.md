# 🪨📄✂️ Shadow Dexterous Hand: Rock-Paper-Scissors

This repository contains the code developed in order to play an interactive Rock-Paper-Scissors game with the Shadow Hand.

## 📌 Project Overview

Shadow Hand Rock Paper Scissors is an advanced implementation of the classic Rock-Paper-Scissors game that supports two modes: 
 1. **Standard Mode**: Play using real-time hand gestures recognition via your webcam
 2. **Robot Mode**: Play against the real robot

The project leverages [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) for gesture recognition, OpenCV for real-time video processing, and integrates with the Shadow Dexterous Hand for robotic movement.

This repository provides the code necessary to set up, control, and monitor the system using **Python** and the **Robotic Operating System (ROS)**.
 - **Robotic Hand**: Shadow Dexterous Hand
 - **Camera**: PC Webcam
 - **Development Environment**: ROS noetic, Docker

## 🖥️ User Interface

![image](https://github.com/user-attachments/assets/d7f4ec6f-ff2e-4268-ae7e-e9484beefc93)

## 🗂️ Folder Structure
 - **[`rock_paper_scissors`](rock_paper_scissors)**: ROS package with the game logic implemented. This package should be placed in the Shadow Hand docker container at `/home/user/projects/shadow_robot/base/src`.

## ⚙️ Software Description
 - [`real_robot.py`](rock_paper_scissors/src/real_robot.py): game script to play with Shadow Hand
   
 - [`only_gui.py`](rock_paper_scissors/src/only_gui.py): game script to play without Shadow Hand
   
 - [`pick_n_place_rs.py`](rock_paper_scissors/scripts/convert_png.py): script to convert images to an adequated format to use with OpenCV
 
 - [`pick_n_place_zed.py`](rock_paper_scissors/scripts/get_shadow_joints.py): script to acquire the current Shadow Hand joint position values


## 🚀 How to Run

### 🦾 Running with Shadow Hand

1. Turn on the Robots
   ⚠️ Ensure that the Shadow Hand’s NUC IP is correctly set.
   
2. Execute `Launch Shadow Right Hand and Arm.desktop`

3. In `Server Docker Container` terminal run `pick_n_place_rs.py`
    ```bash
      roslaunch rock_paper_scissors real_robot.launch
    ```
    
### 🖥️ Running without Shadow Hand

1. Simply run the python script [`only_gui.py`](rock_paper_scissors/src/only_gui.py)

    
## 📫 Contact

Developed by Bruno Santos in DIGI2 Lab

Feel free to reach out via email: brunosantos@fe.up.pt

Last updated in: ``20/11/2024``

