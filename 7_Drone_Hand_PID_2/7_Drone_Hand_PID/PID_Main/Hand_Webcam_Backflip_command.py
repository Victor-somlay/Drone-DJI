import cv2 as cv
import mediapipe as mp
import numpy as np
import Hand_voice_tools_vproto as hvt
from djitellopy import Tello
import time

# Initialisation du drone
drone = Tello()
drone.connect()
drone.streamon()
drone.takeoff()  # Faire décoller le drone

# Initialisation de la webcam
cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

t = 0
while t != 10000:
    success, img = cap.read()
    
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    points = np.zeros([21, 2])
    detection = False
    
    if results.multi_hand_landmarks:
        detection = True
        handLms = results.multi_hand_landmarks[0]
        
        for id, lm in enumerate(handLms.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            points[id, :] = [cx, cy]
            cv.circle(img, (cx, cy), 10, (255, 0, 0), -1)

    if detection:
        etat_dgt = hvt.Etat_dgt(points)  # Détection des doigts
        img = hvt.Tracking_mask(img, points)  # Masque main
        incl = hvt.Incl_compute(points, 13)  # Inclinaison main
        barycentre = hvt.Barycentre(points)  # Barycentre

        # Affichage
        cv.line(img, (int(barycentre[0] - 10000 * hvt.cosinus(incl)), int(barycentre[1] + 10000 * hvt.sinus(incl))),
                (int(barycentre[0] + 10000 * hvt.cosinus(incl)), int(barycentre[1] - 10000 * hvt.sinus(incl))), (0, 255, 0), 2)

        marge = [-30, -40]
        nb_fingers = sum(etat_dgt)  # Compte les doigts levés

        for dgt in range(4, 22, 4):
            dgt_msg = int(dgt / 4 - 1)
            msg = 'Ouvert' if etat_dgt[dgt_msg] == 1 else 'Ferme'
            cv.putText(img, msg, tuple(points[dgt, :].astype(int) + marge), 1, 2, (255, 0, 255), 2)

        # Si deux doigts sont levés, on fait un backflip
        if etat_dgt.tolist() == [1,0,0,0,1] :
            print("Backflip déclenché !")
            drone.flip("b")
            time.sleep(0.5)  # Pause pour éviter des flips répétés

         # Si deux doigts sont levés, on fait un backflip
        if etat_dgt.tolist() == [0,1,1,0,0] :
            print("Backflip déclenché !")
            drone.flip("f")
            time.sleep(0.5)  # Pause pour éviter des flips répétés
        
        if etat_dgt.tolist() == [1,0,0,0,0] :
            print("Backflip déclenché !")
            drone.land()




    cv.imshow("Image", img)
    cv.waitKey(1)
    t += 1

# Fermeture propre
cv.destroyAllWindows()
cap.release()
drone.land()  # Atterrissage du drone
drone.streamoff()
