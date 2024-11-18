import os
import cv2
import time
import random
import numpy as np
import mediapipe as mp

# Images folder path
IMAGES_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGES_PATH = IMAGES_PATH[:-3] + 'images/'

def countFingersUp (mp_hands, result):
    '''
    This function will count the number of fingers up
    Args:
        results: The output of the mediapipe hands landmarks detection
    Returns:
        fingersUp: An array corresponding to fingers position [0-down or 1-up] for each finger by order: ff, mf, rf, lf 
    '''

    # Return array
    fingersUp = [0, 0, 0, 0]

    # Get hand info
    hand_label = result.multi_handedness[0].classification[0].label
    hand_landmarks = result.multi_hand_landmarks[0]

    # Check fingers position
    fingerTipsIds = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    fingersCount = 0
    for tipIndex in fingerTipsIds:
        if (hand_landmarks.landmark[tipIndex].y < hand_landmarks.landmark[tipIndex-2].y):
            fingersUp[fingersCount] = 1
        fingersCount += 1

    return fingersUp


def findGesture (fingersUp):
    '''
    This function will recognize the hand gesture
    Args:
        fingersUp: An array corresponding to fingers position [0-down or 1-up] for each finger by order: thumb, ff, mf, rf, lf 
    Returns:
        gesture: String with 'Rock', 'Paper', 'Scissors' or None
    '''
    gesture = None
    if fingersUp == [0,0,0,0]:
        gesture = 'Rock'
    elif fingersUp == [1,1,1,1]:
        gesture = 'Paper'
    elif fingersUp == [1,1,0,0]:
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
    overlay = cv2.resize(overlay, (150,150), interpolation=cv2.INTER_AREA)
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

    # Init Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Init Mediapipe
    mp_hands = mp.solutions.hands
    detect_hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.75)
    mp_drawing = mp.solutions.drawing_utils
    
    # Game vars
    timer = 0
    initialTime = 0
    startGame = False
    stateResult = False
    userMove = None
    shadowMove = None
    userScore = 0
    shadowScore = 0

    while True:
        # Get background and webcam image
        background = cv2.imread(IMAGES_PATH+"background.png")
        success, raw_img = cap.read()
        
        # Resize webcam image
        img = cv2.resize(raw_img, (0,0), fx=0.885, fy=0.885)
        img = img[:, 138:503]

        # Mediapipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = detect_hands.process(imgRGB)

        # Display Mediapipe Hands
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Game Start
        if startGame:
            if not stateResult:
                timer = time.time() - initialTime + 1
                cv2.putText(background, str(int(timer)), (607,400), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
                # Decision time
                if timer >= 3:
                    if result.multi_hand_landmarks:
                        fingersUp = countFingersUp(mp_hands, result)
                        userMove = findGesture(fingersUp)
                    shadowMove = random.choice(["Rock", "Paper", "Scissors"])
                    stateResult = True
                    startGame = False
                    # Set scores
                    if userMove!=shadowMove:
                        if (userMove=='Rock' and shadowMove=='Scissors') or (userMove=='Paper' and shadowMove=='Rock') or (userMove=='Scissors' and shadowMove=='Paper'):
                            userScore += 1
                        elif (shadowMove=='Rock' and userMove=='Scissors') or (shadowMove=='Paper' and userMove=='Rock') or (shadowMove=='Scissors' and userMove=='Paper') or (userMove==None):
                            shadowScore += 1

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
        elif key == ord('\r') or key == ord('\n'):
            timer = 0
            initialTime = time.time()
            startGame = True
            stateResult = False
            shadowMove = None
            userMove = None


if __name__ == "__main__":
    main()