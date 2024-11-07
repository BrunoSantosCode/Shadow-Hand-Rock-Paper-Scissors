import cv2
import mediapipe as mp



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


def main():

    # Init Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Init Mediapipe
    mp_hands = mp.solutions.hands
    detect_hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.75)
    mp_drawing = mp.solutions.drawing_utils
    

    while True:
        # Get background and webcam image
        background = cv2.imread("rock_paper_scissors/images/background.png")
        success, raw_img = cap.read()
        
        # Resize webcam image
        img = cv2.resize(raw_img, (0,0), fx=0.877, fy=0.877)
        img = img[:, 140:500]

        # Mediapipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = detect_hands.process(imgRGB)

        # Game vars
        shadowGesture = None
        userGesture = None

        # Check if user move
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingersUp = countFingersUp(mp_hands, result)
            userGesture = findGesture(fingersUp)
            if userGesture is not None:
                textSize = cv2.getTextSize(userGesture, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0][0]
                textX = int(962.5 - textSize/2)
                cv2.putText(background, userGesture, (textX,625), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
        
        # Add webcam image to the background
        background[168:589, 783:1143] = img

        # Display
        cv2.imshow("RockPaperScissors.exe", background)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            return


if __name__ == "__main__":
    main()