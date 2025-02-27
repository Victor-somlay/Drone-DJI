#Importations

import cv2 as cv
import mediapipe as mp
from djitellopy import tello
import time
import numpy as np

#Initialisation

drone = tello.Tello()   #appel du drone
drone.connect()   #connexion au drone

drone.send_rc_control(0, 0, 0, 0) # Remet les commandes à zeros
drone.takeoff() # commande de décollage
time.sleep(0.5)
drone.move_up(150)
time.sleep(1)
drone.move_down(100)
drone.land() #commande d'attérissage