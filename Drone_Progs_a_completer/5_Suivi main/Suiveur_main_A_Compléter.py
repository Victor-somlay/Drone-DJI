# Hand tracking avec le drone.
# pour que cela fonctionne il faut que:
# - la batterie soit chargée à fond;
# - la bibliothèque djitellopy soit à jour
# - le drone ne soit pas à contre-jour
# - le câble ethernet soit débranché

# try:
# except KeyboardInterrupt

from djitellopy import Tello
import numpy as np
import cv2
import Drone_Hand as dh
import mediapipe as mp


#import ImageProcessing3L

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

import time

# speed of drone
speed = 20  # 20

# init Tello
tello = Tello()

# tello.get_battery()

# Tello velocities
velocity_fb = 0  # forward/back
velocity_lr = 0  # left/right
velocity_ud = 0  # up/down
velocity_yaw = 0  # yaw

# load cascade classifier 
# Load mediapipe # !!!

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Tello setup

tello.connect()
# tello.set_speed(speed)
tello.streamon()
tello.takeoff()

# init object to read video frames from Tello
frame_read = tello.get_frame_read()

# tello staying still or moving to find target
hold_position = False

running = True

# main loop
while running:

    key = cv2.waitKey(1)

    if key == ord('q'):
        cv2.destroyAllWindows()
        break

    if frame_read.stopped:
        break

    # frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
    frame = np.fliplr(frame_read.frame)
    # image enhancement
    #frame = ImageProcessing3L.enhance(frame)
    frame_shape = frame.shape

    # need a dst object to add shapes/lines on top of the frame, and
    # cv2.cvtColor() returns a dst
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print(type(frame), '   ', frame.shape)
    # recuperation du cadre de la frame
    c, f = dh.hands_mystere(frame)
      
#----------------------A Compléter ---------------------------
    





#----------------------------------------------------------------

    tello.send_rc_control(velocity_lr, velocity_fb, velocity_ud, velocity_yaw)

    # middle of frame 
    # cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 5, (0, 255, 0))
    # cv2.imshow('Tello Video', frame)

    cv2.imshow('Tello Video', frame_annoted)

tello.land()
tello.end()
