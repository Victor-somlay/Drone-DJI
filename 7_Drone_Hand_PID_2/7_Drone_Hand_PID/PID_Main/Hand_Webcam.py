import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import Hand_voice_tools_vproto as hvt
cap = cv.VideoCapture(0)
 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

t=0
while t!=10000:
    success, img = cap.read()
    
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    results = hands.process(imgRGB)
    
    points=np.zeros([21,2])
    detection=False
    
    if results.multi_hand_landmarks:
            detection=True
            handLms = results.multi_hand_landmarks[0]
            
            #################### Détecte ####################
            
            for id, lm in enumerate(handLms.landmark): # Remplie un array des positions des points des mains
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                points[id,:]=[cx,cy]
                
                cv.circle(img,(cx,cy),10,(255,0,0),-1) #Affichage

    if detection:
        etat_dgt= hvt.Etat_dgt(points) # Etat des doigts
        img= hvt.Tracking_mask(img,points) # Masque pour la main
        incl= hvt.Incl_compute(points,13) # Inclinaison de la main
        barycentre= hvt.Barycentre(points) # Barycentre
        
        #Affichage
        cv.line(img,(int(barycentre[0]-10000*hvt.cosinus(incl)),int(barycentre[1]+10000*hvt.sinus(incl))),(int(barycentre[0]+10000*hvt.cosinus(incl)),int(barycentre[1]-10000*hvt.sinus(incl))),(0,255,0),2)
        marge=[-30,-40]
        for dgt in range(4,22,4):
            dgt_msg=int(dgt/4-1)
            if etat_dgt[dgt_msg]==1:
                msg='Ouvert'
            else:
                msg='Ferme'
            cv.putText(img,msg,tuple(points[dgt,:].astype(int)+marge),1,2,(255,0,255),2)
            
    cv.imshow("Image", img)
    cv.waitKey(1)
    #print(t)
    t+=1
    
cv.destroyAllWindows()
cap.release()







