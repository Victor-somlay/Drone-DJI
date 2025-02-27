#face tracking avec le drone.
# pour que cela fonctionne il faut que:
# - la batterie soit chargée à fond;
# - la bibliothèque djitellopy soit à jour
# - le drone ne soit pas à contre-jour
# - le câble ethernet soit débranché

#try:
#except KeyboardInterrupt

from djitellopy import Tello
import numpy as np 
import cv2

# speed of drone 
speed = 25 # 20

# init Tello
tello = Tello()

# Tello velocities
velocity_fb = 0    # forward/back
velocity_lr = 0    # left/right
velocity_ud = 0    # up/down
velocity_yaw = 0   # yaw

# load cascade classifier 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Tello setup
tello.connect()
#tello.set_speed(speed)
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
    frame_shape = frame.shape

    # need a dst object to add shapes/lines on top of the frame, and
    # cv2.cvtColor() returns a dst
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # detect faces
    faces = face_cascade.detectMultiScale(frame, 1.5, 2)

#----------------------------PARTIE A COMPLETER-----------------------------

if len(faces) > 0:
    # Prendre le plus grand visage détecté
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_center_x = x + w // 2
    face_center_y = y + h // 2

    # Centre de l'image
    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2
    desired_face_size = 150  # Taille idéale du visage détecté

    # Calcul des erreurs
    error_x = face_center_x - frame_center_x
    error_y = face_center_y - frame_center_y
    error_size = desired_face_size - w

    # Ajustement de la rotation (Yaw)
    if abs(error_x) > 30:
        velocity_yaw = -int(np.clip(error_x * 0.3, -speed, speed))
    else:
        velocity_yaw = 0

    # Ajustement avant/arrière (distance au visage)
    if abs(error_size) > 30:
        velocity_fb = int(np.clip(error_size * 0.4, -speed, speed))
    else:
        velocity_fb = 0

    # Ajustement haut/bas
    if abs(error_y) > 30:
        velocity_ud = -int(np.clip(error_y * 0.3, -speed, speed))
    else:
        velocity_ud = 0

    # Dessiner un rectangle autour du visage détecté
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

else:
    # Si aucun visage n'est détecté, ne pas bouger
    velocity_fb = 0
    velocity_lr = 0
    velocity_ud = 0
    velocity_yaw = 0



#---------------------------------------------------------------------------    
    tello.send_rc_control(velocity_lr, velocity_fb, velocity_ud, velocity_yaw)

    # middle of frame 
    #cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 5, (0, 255, 0))
    cv2.imshow('Tello Video', frame)

tello.land()
tello.end()