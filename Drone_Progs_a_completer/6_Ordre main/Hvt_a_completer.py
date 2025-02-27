#Importation Benhamou Merwan
import numpy as np
import cv2 as cv
import math
from djitellopy import tello
import time

def Tracking_mask(Shape,ptn):
    # Position masque
    xmax=int(ptn[:,0].max())
    xmin=int(ptn[:,0].min())
    ymax=int(ptn[:,1].max())
    ymin=int(ptn[:,1].min())
    
    # Marge
    mx=int((xmax-xmin)*0.2)+20
    my=int((ymax-ymin)*0.1)+20
    
    # Masque
    mask=np.zeros(Shape)
    cv.rectangle(mask,(xmax+mx,ymin-my),(xmin-mx,ymax+my),(1,1,1),-1)
    mask=mask.astype('uint8')
    
    return mask

def Etat_dgt(ptn):
    '''Calcule le nombre de doigts qui sont ouverts sur une main'''
#--------------------A Compléter----------------------------
    

#-----------------------------------------------------------
    
    return Etat

def Barycentre(ptn):
    return ((ptn[0,:]+ptn[5,:]+ptn[17,:])/3).astype(int)
 
def Detection_Main(results,Shape):   
    points=np.zeros([21,2])
   
    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        for id, lm in enumerate(handLms.landmark): # Remplie un array des positions des points des mains
            h, w, c = Shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            points[id,:] = [cx,cy]
        handtype = results.multi_handedness[0].classidication[0].label
        return points,True, handtype
    else:
        return points,False, None
    
class Pilot:
    
    def __init__(self):
        self.ordre="Aucun"
        # self.niveau_ordre=1 # Niveau permettant la séparation d'action utilisant potentiellement la même configuration de main. Evite le conflit entre plusieurs ordres
        self.atteri=True
        self.altitude=0
        self.clock=[] #capsule contenant le temps et une information
        self.forward=True # Permet qu'il avance une seule fois
        self.temps_confirmation=2
        self.mains=None
        self.left = True
        self.right = True
        self.back = True

    def decision(self,e_d,bar,handType): # Prise de décision en fonctiond de la main
        val_d=len(e_d[e_d==1]) # valeur des doigts (nombres ouverts)
        print(val_d)
        self.ordre="Aucun"
        if val_d==0:
            if len(self.clock)==0: # initialisation 
                    self.clock=[e_d,time.time()]            
            else:
                if (e_d!=self.clock[0]).any(): # redémarre à 0
                    self.clock=[]
                elif time.time()-self.clock[1]>self.temps_confirmation: # délai d'attente (action confirmé !)
                    self.clock=[]
                    self.ordre="Land"   # ordre donné après confirmation

    #------------------------- A compléter ---------------------------------------
                    
    #-----------------------------------------------------------------------------

    def pilot(self,drone): # Commade de pilotage envoyé au drone
        self.altitude=drone.get_distance_tof() # Mesure altitude
        
        # Sécurité Hauteur
        if self.altitude<20 or self.altitude>200 or self.ordre=="Land": # Arret d'urgence au dessus de 2 mètres
            self.ordre="Land"
            print("/!\   -------> Arret d'urgence")
            drone.land()
        
#------------------------A compléter-----------------------------
            

#----------------------------------------------------------------