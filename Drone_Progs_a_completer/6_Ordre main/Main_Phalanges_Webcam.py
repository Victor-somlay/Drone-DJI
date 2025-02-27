# Importation Benhamou Merwan
import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import Hvt_a_completer as hvt

# --------------------------------( Initialisation Drone/[Webcam])---------------------------------------------
#from djitellopy import tello
#drone = tello.Tello()  # appel du drone
#drone.connect()  # la connexion du drone
#drone.streamon()  # activer le flux vidéo
####                  avant: Drone //////////////////// après: Webcam

cap = cv.VideoCapture(0)

pilot = hvt.Pilot()

# --------------------------------( Initialisation de médiapipe )-------------------------------------------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=8)  # Création de l'objet objet de classe mediapipe
# hands = mpHands.Hands()  # Création de l'objet objet de classe mediapipe
mp_drawing = mp.solutions.drawing_utils  # Permet l'affichage des points de la main (voir site)
mp_hands = mp.solutions.hands

start_time = time.time()
success, img = cap.read()


fps = []
t = 0
while t != 1000 :  # t pour arreter le programme au bout d'un certain temps

    # ------------------------------------( Partie Traitement Images )---------------------------------------------------------------

    #              ##########################[ Capture image ]###########################

    success, img = cap.read() # Capture Webcam
    masque = np.ones(img.shape, dtype='uint8')
    imgRGB = cv.cvtColor(img * masque, cv.COLOR_BGR2RGB)

    #              ##########################[ Capture Main ]###########################

    results = hands.process(imgRGB)
    #print(results)

    points, detection = hvt.Detection_Main(results, img.shape)  # Détecte la position des phalanges
    print(points)

    #            ##################[ Détermination caractéristique Main ]###################

    if detection:
        etat_dgt = hvt.Etat_dgt(points)  # Etat des doigts (ouvert/fermé)
        # masque = hvt.Tracking_mask(img.shape,points) # Masque inverse pour la main (tout est caché sauf celle-ci, cela permet de suivre toujours la même main)
        barycentre = hvt.Barycentre(points)  # Barycentre

    # ------------------------------------( Pilotage )---------------------------------------------------------------

    # ------------------------------------( Affichage )---------------------------------------------------------------
    #img[0:10, :] = 0

    #img[11:23, :] = (0, 0, 255)
    if detection:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #mp_drawing.draw_landmarks( )
            #print(hand_landmarks)

    # Calcul fps
    # fps=0
    # start_time=time.time()
    # cv.putText(img,"Ordre : "+pilot.ordre,(20,30),1,1.2,(255,255,255),1)
    # cv.putText(img,"Altitude : "+str(pilot.altitude)+"      fps : "+str(int(np.mean(fps[-10:]))),(250,30),1,1.2,(255,255,255),1)
    cv.imshow("Image", img)
    cv.waitKey(1)
    t += 1

cv.destroyAllWindows()
cap.release()




