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
speed = 30  # 20
error = np.zeros(4)
perror = np.zeros(4)
ierror = np.zeros(4)
pid = [0.4, 0.1, 0.4]

# init Tello


# Tello velocities
velocity_fb = 0  # forward/back
velocity_lr = 0  # left/right
velocity_ud = 0  # up/down
velocity_yaw = 0  # yaw


# init object to read video frames from Tello
frame_read = cv2.VideoCapture(0)

# tello staying still or moving to find target
hold_position = False

running = True

# main loop
while running:

    key = cv2.waitKey(1)

    if key == ord('q'):
        cv2.destroyAllWindows()
        break



    # frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
    success, image = frame_read.read()
    frame = np.fliplr(image)
    # image enhancement
    #frame = ImageProcessing3L.enhance(frame)
    frame_shape = frame.shape

    # need a dst object to add shapes/lines on top of the frame, and
    # cv2.cvtColor() returns a dst
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print(type(frame), '   ', frame.shape)
    # recuperation du cadre de la frame
    cadre, frame_annoted = dh.hands_cadre(frame)
    # detect HANDS
    if len(cadre)==0:
        x,y,w,h = 0,0,0,0
    else:
        x, y, w, h = cadre

    hand_middle = (x + w // 2, y + h // 2)  # middle of hand coordinate
    frame_middle = (frame_shape[1] // 2, frame_shape[0] // 2)  # middle of frame coordinate

    # want face to be around 200x200 pixels
    # if face is larger, then both dimensions of z_axis_displacement
    # are negative, so we can tell the drone to move backwards because it
    # is too close to the face
    z_axis_displacement = (200 - w, 200 - h)

    # displacement of face from middle of frame; yaw is 0; NOTE: using only width of face for z axis displacement
    displacement_vector = np.array(
        [frame_middle[0] - hand_middle[0], z_axis_displacement[0], frame_middle[1] - hand_middle[1], 0])

    print(displacement_vector)

    error = displacement_vector
    ierror += displacement_vector
    # threshold circle
    # cv2.circle(frame, (face_middle[0], face_middle[1]), 200, (255, 255, 255), 5)

    # vector line
    # cv2.line(frame, (x + w//2 , y + h//2), (x + w//2 + displacement_vector[0] , y + h//2 + displacement_vector[2]), (255, 255, 255), 2)

    # Modification ici
    displacement_vector_pid_yaw_tanh = np.tanh(
        (pid[0] * error[0] + pid[1] * ierror[0] + pid[2] * (error[0] - perror[0])) / 400)
    displacement_vector_pid_yaw_pid = (pid[0] * error[0] + pid[1] * ierror[0] + pid[2] * (error[0] - perror[0])) / 2000

    if not abs(displacement_vector[0]) < 50:
        velocity_yaw = int(speed * (displacement_vector_pid_yaw_tanh / 2 + displacement_vector_pid_yaw_pid / 2))

    else:
        velocity_yaw = 0
        ierror[0] = 0
        error[0] = 0

    displacement_vector_pid_fb_tanh = np.tanh(
        (pid[0] * error[1] + pid[1] * ierror[1] + pid[2] * (error[1] - perror[1])) / 200)
    displacement_vector_pid_fb_pid = (pid[0] * error[1] + pid[1] * ierror[1] + pid[2] * (error[1] - perror[1])) / 1200
    if not abs(displacement_vector[1]) < 40:
        velocity_fb = int(
            speed / 2 * (2 / 3 * displacement_vector_pid_fb_tanh + 1 / 3 * displacement_vector_pid_fb_pid))
    else:
        velocity_fb = 0
        ierror[1] = 0
        error[1] = 0

    displacement_vector_pid_ud_tanh = np.tanh(
        (pid[0] * error[2] + pid[1] * ierror[2] + pid[2] * (error[2] - perror[2])) / 600)
    displacement_vector_pid_ud_pid = (pid[0] * error[2] + pid[1] * ierror[2] + pid[2] * 2 * (
                error[2] - perror[2])) / 1500
    if not abs(displacement_vector[2]) < 40:
        velocity_ud = int(speed * (2 * displacement_vector_pid_ud_tanh / 2 + 0 * displacement_vector_pid_ud_pid / 2))
    else:
        velocity_ud = 0
        ierror[2] = 0
        error[2] = 0

        # break for loop because we only want to move to the first face detected
    if len(cadre) == 0:
        velocity_fb = 0
        velocity_lr = 0
        velocity_ud = 0
        velocity_yaw = 0
        error = np.zeros(4)
        perror = np.zeros(4)
        ierror = np.zeros(4)

    print(velocity_lr)
    print(velocity_fb)
    print(velocity_ud)
    print(velocity_yaw)

    dh.show_velocities_on_image(frame_annoted, velocity_lr, velocity_ud, velocity_fb, velocity_yaw)

    #cv2.putText(frame_annoted, "velocity_lr :" + str(velocity_lr) + "\nvelocity_fb :" + str(velocity_fb) + "\nvelocity_ud :" +str(velocity_ud)+"\nvelocity_yaw :" +str(velocity_yaw), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    # middle of frame
    # cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 5, (0, 255, 0))
    # cv2.imshow('Tello Video', frame)

    cv2.imshow('Tello Video', frame_annoted)
    perror = error


