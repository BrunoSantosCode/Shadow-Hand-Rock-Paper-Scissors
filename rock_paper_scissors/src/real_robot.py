#!/usr/bin/env python3

import os
from tracemalloc import start
import cv2
import rospy
import random
import numpy as np
import mediapipe as mp
from termcolor import colored
from threading import Thread, Event
from mediapipe.framework.formats import landmark_pb2
from sr_robot_commander.sr_hand_commander import SrHandCommander

# Shadow Hand Poses

# Default
default_hand = {'rh_FFJ1': 1.571, 'rh_FFJ2': 1.571, 'rh_FFJ3': 1.571, 'rh_FFJ4': 0.000,
                'rh_LFJ1': 1.571, 'rh_LFJ2': 1.571, 'rh_LFJ3': 1.571, 'rh_LFJ4': 0.000, 'rh_LFJ5': 0.000,
                'rh_MFJ1': 1.571, 'rh_MFJ2': 1.571, 'rh_MFJ3': 1.571, 'rh_MFJ4': 0.000,
                'rh_RFJ1': 1.571, 'rh_RFJ2': 1.571, 'rh_RFJ3': 1.571, 'rh_RFJ4': 0.000,
                'rh_WRJ1': 0.000, 'rh_WRJ2': 0.000}
default_hand_th = {'rh_THJ1': 0.716, 'rh_THJ2': 0.691, 'rh_THJ3': 0.201, 'rh_THJ4': 1.226, 'rh_THJ5': 0.186}

# Open Thumb
open_thumb = {'rh_THJ1': 0.150, 'rh_THJ2': 0.200, 'rh_THJ3': 0.200, 'rh_THJ4': 1.200, 'rh_THJ5': -0.100}

# Rock
Rock_hand = {'rh_FFJ1': 1.571, 'rh_FFJ2': 1.571, 'rh_FFJ3': 1.571, 'rh_FFJ4': 0.000,
             'rh_LFJ1': 1.571, 'rh_LFJ2': 1.571, 'rh_LFJ3': 1.571, 'rh_LFJ4': 0.000, 'rh_LFJ5': 0.000,
             'rh_MFJ1': 1.571, 'rh_MFJ2': 1.571, 'rh_MFJ3': 1.571, 'rh_MFJ4': 0.000,
             'rh_RFJ1': 1.571, 'rh_RFJ2': 1.571, 'rh_RFJ3': 1.571, 'rh_RFJ4': 0.000,
             'rh_THJ1': 0.730, 'rh_THJ2': 0.698, 'rh_THJ3': 0.000, 'rh_THJ4': 1.221, 'rh_THJ5': 0.050,
             'rh_WRJ1': 0.000, 'rh_WRJ2': 0.000}

# Paper
Paper_hand = {'rh_FFJ1': 0.000, 'rh_FFJ2': 0.000, 'rh_FFJ3': 0.000, 'rh_FFJ4': 0.000, 
              'rh_LFJ1': 0.000, 'rh_LFJ2': 0.000, 'rh_LFJ3': 0.000, 'rh_LFJ4': 0.000, 'rh_LFJ5': 0.000,
              'rh_MFJ1': 0.000, 'rh_MFJ2': 0.000, 'rh_MFJ3': 0.000, 'rh_MFJ4': 0.000,
              'rh_RFJ1': 0.000, 'rh_RFJ2': 0.000, 'rh_RFJ3': 0.000, 'rh_RFJ4': 0.000,
              'rh_THJ1': 0.000, 'rh_THJ2': 0.000, 'rh_THJ3': 0.000, 'rh_THJ4': 0.000, 'rh_THJ5': 0.000,
              'rh_WRJ1': 0.000, 'rh_WRJ2': 0.000}

# Scissor
Scissors_hand = {'rh_FFJ1': 0.000, 'rh_FFJ2': 0.000, 'rh_FFJ3': 0.000, 'rh_FFJ4': -0.180,
                'rh_LFJ1': 1.571, 'rh_LFJ2': 1.571, 'rh_LFJ3': 1.571, 'rh_LFJ4': 0.000, 'rh_LFJ5': 0.000,
                'rh_MFJ1': 0.000, 'rh_MFJ2': 0.000, 'rh_MFJ3': 0.000, 'rh_MFJ4': 0.180,
                'rh_RFJ1': 1.571, 'rh_RFJ2': 1.571, 'rh_RFJ3': 1.571, 'rh_RFJ4': 0.000,
                'rh_THJ1': 0.730, 'rh_THJ2': 0.698, 'rh_THJ3': 0.000, 'rh_THJ4': 1.221, 'rh_THJ5': 0.050,
                'rh_WRJ1': 0.000, 'rh_WRJ2': 0.000}

# Default Wrist
wrist_default = {'rh_WRJ1': 0.000, 'rh_WRJ2': 0.000}

# Wrist Up
wrist_up = {'rh_WRJ1': 0.000, 'rh_WRJ2': 0.174}

# Wrist Down
wrist_down = {'rh_WRJ1': 0.000, 'rh_WRJ2': -0.523}


def set_hand_pose(pose: str, wait = True):
    """
        Sets a pre-defined pose for Shadow Hand
        @param pose - A given object 'default', 'Rock', 'Paper' or 'Scissors'
        @param wait - Set a blocking or non-blocking command
    """
    global hand_commander

    if (pose == 'default'):
        hand_joints = hand_commander.get_joints_position()
        sum_ff = (hand_joints['rh_FFJ2']+hand_joints['rh_FFJ3'])*180/3.1415
        sum_mf = (hand_joints['rh_MFJ2']+hand_joints['rh_MFJ3'])*180/3.1415
        if ((sum_ff < 120) or (sum_mf < 120)): 
            hand_commander.move_to_joint_value_target_unsafe(joint_states=open_thumb, time=0.30, wait=True, angle_degrees=False)
        hand_commander.move_to_joint_value_target_unsafe(joint_states=default_hand, time=0.30, wait=wait, angle_degrees=False)
        hand_commander.move_to_joint_value_target_unsafe(joint_states=default_hand_th, time=0.30, wait=wait, angle_degrees=False)
    elif pose == 'Rock':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=Rock_hand, time=0.30, wait=wait, angle_degrees=False)
    elif pose == 'Paper':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=Paper_hand, time=0.30, wait=wait, angle_degrees=False)
    elif pose == 'Scissors':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=open_thumb, time=0.30, wait=True, angle_degrees=False)
        hand_commander.move_to_joint_value_target_unsafe(joint_states=Scissors_hand, time=0.30, wait=wait, angle_degrees=False)
    else:
        print('\n' + colored('ERROR: "' + pose + '" hand pose is not defined!', 'red') + '\n')


def set_wrist_pose(pose: str, wait = True):
    """
        Sets a pre-defined pose for Shadow Hand wrist
        @param pose - A given pose 'up' or 'down'
        @param wait - Set a blocking or non-blocking command
    """
    global hand_commander

    if pose == 'default':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=wrist_default, time=0.30, wait=wait, angle_degrees=False)
    elif pose == 'up':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=wrist_up, time=0.45, wait=wait, angle_degrees=False)
    elif pose == 'down':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=wrist_down, time=0.45, wait=wait, angle_degrees=False)
    else:
        print('\n' + colored('ERROR: "' + pose + '" wrist pose is not defined!', 'red') + '\n')


def give_me_a(pose: str, flags):
    """
        Play Shadow Hand
        @param pose - A given object 'default', 'Rock', 'Paper' or 'Scissors'
        @param wait - Set a blocking or non-blocking command
    """
    try:
        set_hand_pose('default', wait=True)
        set_wrist_pose('up', wait=True)
        set_wrist_pose('down', wait=True)
        flags['wrist1'].set()
        set_wrist_pose('up', wait=True)
        set_wrist_pose('down', wait=True)
        flags['wrist2'].set()
        set_wrist_pose('up', wait=True)
        set_wrist_pose('default', wait=True)
        flags['wrist3'].set()
        set_hand_pose(pose=pose, wait=True)
    finally:
        flags['complete'].set()


# Images folder path
IMAGES_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGES_PATH = IMAGES_PATH[:-3] + 'images/'

# MediaPipe Gesture Recognition model path
MODEL_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = MODEL_PATH[:-3] + 'models/gesture_recognizer.task'


def getUserMove(mp_result):
    '''
    This function will return the user move from the MediaPipe results
    Args:
        mp_result: Hand gesture recognition from MediaPipe
    Returns:
        gesture: String with 'Rock', 'Paper', 'Scissors' or None
    '''
    gesture = None
    if mp_result == 'Closed_Fist':
        gesture = 'Rock'
    elif mp_result == 'Open_Palm':
        gesture = 'Paper'
    elif mp_result == 'Victory':
        gesture = 'Scissors'

    return gesture

def overlay_image(background, overlay, center_position):
    """
    Overlays a smaller image (overlay) onto a background image at a specified center position.
    
    Args:
        background (numpy.ndarray): The background image.
        overlay (numpy.ndarray): The overlay image with an alpha channel (transparency).
        center_x (int): X-coordinate of the center position for the overlay.
        center_y (int): Y-coordinate of the center position for the overlay.
        
    Returns:
        numpy.ndarray: The background image with the overlay applied.
    """
    # Unpack center position coordinates
    center_x, center_y = center_position
    # Resize the overlay image to the specified size
    overlay = cv2.resize(overlay, (275,367), interpolation=cv2.INTER_AREA)
    # Get dimensions of the overlay image
    overlay_height, overlay_width = overlay.shape[:2]
    # Calculate the top-left corner of the overlay based on the center position
    top_left_x = center_x - overlay_width // 2
    top_left_y = center_y - overlay_height // 2
    # Check if overlay is within bounds
    if (top_left_x < 0 or top_left_y < 0 or
        top_left_x + overlay_width > background.shape[1] or
        top_left_y + overlay_height > background.shape[0]):
        raise ValueError("Overlay image exceeds background boundaries at the specified center position.")
    # Separate the color and alpha channels of the overlay
    overlay_rgb = overlay[:, :, :3]  # Color channels (BGR)
    overlay_alpha = overlay[:, :, 3]  # Alpha channel
    # Crop the overlay section in the background where the overlay will be placed
    background_section = background[top_left_y:top_left_y+overlay_height, top_left_x:top_left_x+overlay_width]
    # Blend overlay with the background section using the alpha mask
    alpha_factor = overlay_alpha[:, :, np.newaxis] / 255.0  # Normalize alpha channel to range [0, 1]
    blended_section = background_section * (1 - alpha_factor) + overlay_rgb * alpha_factor
    # Place the blended section back into the background image
    background[top_left_y:top_left_y+overlay_height, top_left_x:top_left_x+overlay_width] = blended_section

    return background



def main():
    global hand_commander

    # Init ROS
    rospy.init_node('rock_paper_scissors')

    # Shadow Hand commander
    hand_commander = SrHandCommander(name='right_hand')

    # Set control velocity and acceleration
    hand_commander.set_max_velocity_scaling_factor(1.0)
    hand_commander.set_max_acceleration_scaling_factor(1.0)

    print('\n' + colored('"rock_paper_scissors" ROS node is ready!', 'green') + '\n') 

    # Init Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Init Mediapipe
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=1)
    recognizer = GestureRecognizer.create_from_options(options)
    
    # Game vars
    startGame = False
    stateResult = False
    userGesture = None
    userMove = None
    shadowMove = None
    userScore = 0
    shadowScore = 0
    thumbsUp = False

    # Flags and process for command Shadow Hand
    flags = {
        'wrist1': Event(),
        'wrist2': Event(),
        'wrist3': Event(),
        'complete': Event()
    }
    flags['complete'].set()
    commandShadow = None

    while True:
        # Get background and webcam image
        background = cv2.imread(IMAGES_PATH+"background.png")
        success, raw_img = cap.read()
        
        # Image Preprocessing
        img = cv2.resize(raw_img, (0,0), fx=0.885, fy=0.885)
        img = img[:, 138:503]
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Mediapipe
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB.astype(np.uint8))
        recognition_result = recognizer.recognize(mp_img)
        if recognition_result.gestures:
            userGesture = recognition_result.gestures[0][0].category_name
            thumbsUp = (userGesture == 'Thumb_Up')

       # Display Mediapipe Hands
        if recognition_result.hand_landmarks:
             for hand_landmarks in recognition_result.hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                mp_drawing.draw_landmarks(img, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS)
        
        # Start Game
        if startGame:
            if flags['complete'].is_set():
                startGame = False
                # Set scores
                if userMove!=shadowMove:
                    if (userMove=='Rock' and shadowMove=='Scissors') or (userMove=='Paper' and shadowMove=='Rock') or (userMove=='Scissors' and shadowMove=='Paper'):
                        userScore += 1
                    elif (shadowMove=='Rock' and userMove=='Scissors') or (shadowMove=='Paper' and userMove=='Rock') or (shadowMove=='Scissors' and userMove=='Paper') or (userMove==None):
                        shadowScore += 1
            elif flags['wrist3'].is_set():
                cv2.putText(background, '3', (607,400), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
                if recognition_result.gestures:
                        userMove = getUserMove(userGesture)
                stateResult = True
            elif flags['wrist2'].is_set():
                cv2.putText(background, '2', (607,400), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
            elif flags['wrist1'].is_set():    
                cv2.putText(background, '1', (607,400), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
            else:
                cv2.putText(background, '0', (607,400), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
            
        # Display results
        if stateResult:
            # Timer
            cv2.putText(background, '3', (607,400), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
            # User Move
            if userMove is not None:
                textSize = cv2.getTextSize(userMove, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
                textX = int(962 - textSize/2)
                cv2.putText(background, userMove, (textX,625), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            else:
                textSize = cv2.getTextSize('Too late!', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
                textX = int(962 - textSize/2)
                cv2.putText(background, 'Too late!', (textX,625), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            # Shadow Move
            shadowMoveImg = cv2.imread(IMAGES_PATH+f'{shadowMove}.png', cv2.IMREAD_UNCHANGED)
            background = overlay_image(background, shadowMoveImg, (318, 377))
            textSize = cv2.getTextSize(shadowMove, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
            textX = int(318 - textSize/2)
            cv2.putText(background, shadowMove, (textX,625), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Display Scores
        offset=4
        if userScore == shadowScore:
            cv2.putText(background, f"{shadowScore:04d}", (536+offset,286-offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(background, f"{userScore:04d}", (642+offset,286-offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        elif userScore > shadowScore:
            cv2.putText(background, f"{shadowScore:04d}", (536+offset,286-offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(background, f"{userScore:04d}", (642+offset,286-offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        elif shadowScore > userScore:
            cv2.putText(background, f"{shadowScore:04d}", (536+offset,286-offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(background, f"{userScore:04d}", (642+offset,286-offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Add webcam image to the background
        background[165:590, 780:1145] = img

        # Display
        cv2.imshow("RockPaperScissors.exe", background)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return
        elif (key == ord('\r') or key == ord('\n') or thumbsUp) and flags['complete'].is_set():
            startGame = True
            stateResult = False
            shadowMove = random.choice(["Rock", "Paper", "Scissors"])
            userMove = None
            thumbsUp = False
            # Clear Flags
            for flag in flags.values():
                flag.clear()
            # Start Thread
            commandShadow = Thread(target=give_me_a, args=(shadowMove, flags))
            commandShadow.start()


if __name__ == "__main__":
    main()