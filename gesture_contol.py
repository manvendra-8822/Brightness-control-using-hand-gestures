import cv2 
import mediapipe as mp 
from math import hypot 
import screen_brightness_control as sbc 
import numpy as np 

# Initialize mediapipe Hands model
mp_hands = mp.solutions.hands 
hands_model = mp_hands.Hands( 
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75, 
    max_num_hands=1) 
  
mp_drawing = mp.solutions.drawing_utils 

# Start capturing video from webcam 
cap = cv2.VideoCapture(0) 

while True: 
    ret, frame = cap.read() 
    if not ret:
        break
  
    # Flip the image 
    frame = cv2.flip(frame, 1) 
  
    # Convert BGR image to RGB image 
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    
    hands_process = hands_model.process(rgb_img) 
  
    landmark_list = [] 
    if hands_process.multi_hand_landmarks: 
        for hand_landmark in hands_process.multi_hand_landmarks: 
            for idx, landmarks in enumerate(hand_landmark.landmark): 
                height, width, color_channels = frame.shape 
  
                x, y = int(landmarks.x*width), int(landmarks.y*height) 
                landmark_list.append([idx, x, y]) 
  
            # Draw landmarks 
            mp_drawing.draw_landmarks(frame, hand_landmark, 
                                       mp_hands.HAND_CONNECTIONS) 
  
    # If landmarks list is not empty 
    if landmark_list: 
        x1, y1 = landmark_list[4][1], landmark_list[4][2] 
        x2, y2 = landmark_list[8][1], landmark_list[8][2] 
        cv2.circle(frame, (x1, y1), 7, (255,192,203), cv2.FILLED) 
        cv2.circle(frame, (x2, y2), 7, (222, 49, 99), cv2.FILLED) 
        cv2.line(frame, (x1, y1), (x2, y2), (255,192,203), 3) 
        length = hypot(x2-x1, y2-y1) 
        brightness_level = np.interp(length, [15, 220], [0, 100]) 
  
        # Set brightness 
        sbc.set_brightness(int(brightness_level)) 
  
    # Display video and break loop on pressing 'q' key
    cv2.imshow('Hand Tracking', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()


'''from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume
import threading

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume = cast(interface,POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVolume = volRange[0]
maxVolume = volRange[1]

def setVolume(dist):
    vol = np.interp(int(dist), [35, 215], [minVolume, maxVolume])
    volume.SetMasterVolumeLevel(vol, None)
    
This is additional code for using the volume control functionality as well
'''
