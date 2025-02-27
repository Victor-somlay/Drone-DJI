# En haut à droite: Edit configurations
# il faut mettre au niveau de 'Script Path' le nom du FICHIER .py
# que l'on veut faire tourner

# Run>Run

# Il faut éviter les contre-jours: le visage doit être bien éclairé et le drone ne
# doit pas être ébloui.

# pour quitter: on clique sur la vidéo et on appuie sur la touche q.
# pour charger: il faut que le drone soit éteint. on appuie brièvement sur le bouton ON/OFF.

# Les problèmes qui peuvent arriver:
# batterie déchargée: ça clignote en rouge. Si le drone est en vol il se pose.
# Si le drone est au sol il ne décolle pas, il affiche juste la vidéo.
import cv2
import numpy as np
import time

from djitellopy import tello

drone1 = tello.Tello()
drone1.connect()
print(drone1.get_battery())

drone1.streamon()
drone1.takeoff()
drone1.send_rc_control(0, 0, 25, 0)
time.sleep(2.2)
pid = [0.4, 0.4, 0]

pError = 0
w,h = 368, 240
fbRange = [6200,6800]

def detectObjet(img):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cx = x+ w//2
        cy = y+ h//2
        area = w*h
        cv2.circle(img, (cx,cy), 5, (0,255,0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


def trackFace( info, w, pid, pError):
    area = info[1]
    x,y  = info[0]
    fb = 0
    error = x-w//2
    speed = pid[0]*error + pid[1]*(error-pError)
    speed = int(np.clip(speed, -100, 100))
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    if area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    #print(speed, fb)
    if x == 0:
        speed = 0
        error = 0
    drone1.send_rc_control(0, fb, 0, speed)
    return error
#cap = cv2.VideoCapture(0)

while True:
    #_, img = cap.read()
    img = drone1.get_frame_read().frame
    img = cv2.resize(img, (w,h))
    img, info = detectObjet(img)
    pError = trackFace( info, w, pid, pError)
    #print("Area", info[1], "Center", info[0])
    detectObjet(img)
    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone1.land()
        break