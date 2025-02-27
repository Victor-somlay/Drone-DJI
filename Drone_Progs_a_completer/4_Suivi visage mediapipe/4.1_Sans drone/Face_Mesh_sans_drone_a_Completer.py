from time import sleep, time

print(" Chargement de l'import...")
t0 = time()

import cv2
import mediapipe as mp
from utils_holistic import *


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh_lips_positions , face_oval_landmarks_positions = get_face_mesh_landmarks_positions()

print(" Fin du chargement de l'import", '\n', "Durée :", time()-t0, "seconde(s)")

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

# Drone speed
speed = 25
# Distance from screen reference
distance_ref = 60 #cm
# Speed limit
max_speed = 40


#Number of detections
n_detection = 0
n_total_frames = 0

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh: #initial confidences = 0.5
  
  while cap.isOpened():
    n_total_frames += 1 
      
    success, image = cap.read()
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    brightness(image)
    
    
#-------------------A Compléter-----------------------



    if ???:
      
        
      for ??? in ????:
          
        
#-----------------------------------------------------

      velocity_lr,velocity_fb,velocity_ud,velocity_yaw = drone_velocity2(image.shape, head_barycenter, nose_coord, distance2screen)
        
    else:
      print('No face detected')
      
      velocity_lr, velocity_fb, velocity_ud, velocity_yaw = 0, 0, 0, 0
      
    show_velocities_on_image(image, velocity_lr, velocity_ud, velocity_fb, velocity_yaw)
    
    
    # Check if there is no overspeed
    if not velocities_inrange(velocity_lr, velocity_ud, velocity_fb, velocity_yaw, max_speed):
      print(f'WARNING, max velocity {max_speed} reached')
      
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', image)
    
    if cv2.waitKey(5) & 0xFF == 27: #Echap
      break
      
# Frame number and detection ratio value 
print('\n')
print("Total frames number: ", n_total_frames)    
print("Detection Ratio: ", n_detection/n_total_frames)


cap.release()



## Implémentation du ratio de détection OK

## Instruction "No face detected" OK

## Implémentation du calcul de distance par rapport à l'écran OK

## Calcul de luminance en temps réel OK

## Arriver à récupérer la ligne verticale du face mesh pour récupérer le point du nez OK

## Algo d'égalisation d'histogrammes - https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

## Courbes de reconnaissance par rapport à la luminance

## Comparaison par rapport au barycentre pour le drône 

## Drône qui se recentre par rapport à l'orientation de la tête