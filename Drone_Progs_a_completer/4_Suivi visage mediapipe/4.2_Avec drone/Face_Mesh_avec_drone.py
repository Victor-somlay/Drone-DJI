from time import sleep, time

print(" Loading imports...")
t0 = time()

import cv2
import mediapipe as mp
from utils_holistic import *
from djitellopy import Tello

## Mp definition
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh_lips_positions, face_oval_landmarks_positions = get_face_mesh_landmarks_positions()
image_shape = (960, 720)

# Drone speed limit
max_speed = 50

# init Tello
tello = Tello()

# load cascade classifier 
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print(" Packages imported", '\n', "Duration :", time() - t0, "second(s)")
print(" Drone is taking off ...")

# Tello setup
tello.connect()
print(" Tello battery =", tello.get_battery())
tello.set_speed(20)
tello.streamon()
tello.takeoff()

# Init object to read video frames from Tello
frame_read = tello.get_frame_read()
video_capture = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, image_shape)

# Number of detections
n_detection = 0
n_total_frames = 0

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:  # initial confidences = 0.5

    while True:
        n_total_frames += 1

        # Obtaining video
        image = np.fliplr(frame_read.frame)

        # Calculate brightness
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        brightness(image)

        # Calculating face recognition
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            n_detection += 1

            for face_landmarks in results.multi_face_landmarks:
                head_barycenter, nose_coord, distance2screen = detect_N_show(image, face_landmarks,
                                                                             mp_face_mesh.FACEMESH_FACE_OVAL,
                                                                             face_mesh_lips_positions,
                                                                             face_oval_landmarks_positions)

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())

                # mp_drawing.draw_landmarks(
                #    image=image,
                #    landmark_list=face_landmarks,
                #    connections=mp_face_mesh.FACEMESH_CONTOURS,
                #    landmark_drawing_spec=None,
                #    connection_drawing_spec=mp_drawing_styles
                #    .get_default_face_mesh_contours_style())

                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_IRISES,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_iris_connections_style())

            # Update velocities
            velocity_lr, velocity_fb, velocity_ud, velocity_yaw = drone_velocity2(image.shape, head_barycenter,
                                                                                  nose_coord, distance2screen)
            if velocity_fb>max_speed:
                velocity_fb=max_speed

            if velocity_fb<-max_speed:
                velocity_fb=-max_speed

            print(velocity_fb)

        else:
            print('No face detected')

            velocity_lr, velocity_fb, velocity_ud, velocity_yaw = 0, 0, 0, 0

        # Show velocities on image
        show_velocities_on_image(image, velocity_lr, velocity_ud, velocity_fb, velocity_yaw)

        # Saving video
        video_capture.write(image)
        print("Frame ajoutée à la vidéo :")

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', image)

        # Check if there is no overspeed
        if velocities_inrange(velocity_lr, velocity_ud, velocity_fb, velocity_yaw, max_speed):
            # Sending velocities value to drone
            tello.send_rc_control(velocity_lr, velocity_fb, velocity_ud, velocity_yaw)
        else:
            print(f'WARNING, max velocity {max_speed} reached')

        if cv2.waitKey(5) & 0xFF == 27:  # Echap
            break

# Stopping video
video_capture.release()

# Closes all the frames
cv2.destroyAllWindows()

# Frame number and detection ratio value 
print('\n')
print("Total frames number: ", n_total_frames)
print("Detection Ratio: ", n_detection / n_total_frames)

print(" Tello battery =", tello.get_battery())

# Landing
tello.land()
tello.end()

## Adapter le yaw et enlever le mouvement left right
