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
    
    selec_doigts=[False,False,False,True,True,False,False,True,True,False,False,True,True,False,False,True,True,True,False,False,True] # Selectionne les deux points des bouts des doigts
    selec_bout=[False,True,False,True,False,True,False,True,False,True] # Selectionne dans l'array du bout des doigts
    selec_phal=[True,False,True,False,True,False,True,False,True,False] # Selectionne dans l'array de la derniere phalanges des doigts
    
    barycentre=((ptn[0,:]+ptn[5,:]+ptn[17,:])/3).astype(int) # Centre paume de la main

    
    Dist_ptn=ptn-barycentre 
    Dist_ptn=np.sqrt(Dist_ptn[:,0]**2+Dist_ptn[:,1]**2) # Distance points/barycentre
    
    Selection=Dist_ptn[selec_doigts] # Permet la selection des points souhaités
    Comp_dist=Selection[selec_bout]-Selection[selec_phal] # comparaison distance bout des doigts/phalanges
    
    Etat=(((Comp_dist/abs(Comp_dist))+1)/2).astype(int) # valeur final 0 ou 1.
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
        if val_d==1:
            if len(self.clock)==0: # initialisation 
                    self.clock=[e_d,time.time()]            
            else:
                if (e_d!=self.clock[0]).any(): # redémarre à 0
                    self.clock=[]
                elif time.time()-self.clock[1]>self.temps_confirmation: # délai d'attente (action confirmé !)
                    self.clock=[]
                    if handType == "Left":
                        self.ordre="Avance"    # ordre donné après confirmation
                    else:
                        self.ordre = "Recule"
                    
        if val_d==3:
            if len(self.clock)==0: # initialisation 
                    self.clock=[e_d,time.time()]            
            else:
                if (e_d!=self.clock[0]).any(): # redémarre à 0
                    self.clock=[]
                elif time.time()-self.clock[1]>self.temps_confirmation: # délai d'attente (action confirmé !)
                    self.clock=[]
                    if handType == "Right":
                        self.ordre="Droite"    # ordre donné après confirmation
                    else:
                        self.ordre = "Gauche"

        if val_d == 5:
            if len(self.clock) == 0:  # initialisation
                self.clock = [e_d, time.time()]
            else:
                if (e_d != self.clock[0]).any():  # redémarre à 0
                    self.clock = []
                elif time.time() - self.clock[1] > self.temps_confirmation:  # délai d'attente (action confirmé !)
                    self.clock = []
                    if handType == "Right":
                        self.ordre = "Down"  # ordre donné après confirmation
                    else:
                        self.ordre = "Up"

    def pilot(self,drone): # Commade de pilotage envoyé au drone
        self.altitude=drone.get_distance_tof() # Mesure altitude
        
        # Sécurité Hauteur
        if self.altitude<20 or self.altitude>200: # Arret d'urgence au dessus de 2 mètres
            self.ordre="Land"
            print("/!\   -------> Arret d'urgence")
            drone.land()
        
        if self.ordre=='Down':
            drone.move_down(abs(100-self.altitude)) # Amène à l'altitude souhaitée = 35 fois la valeur indiquée, nombre doigts levé=[0-5] >>> altitude=[0-175]

        if self.ordre=='Up':
            drone.move_up(abs(150-self.altitude))

        if self.ordre == 'Droite' and self.right:
            drone.move_left(100)  # Fait aller à droite le drone
            if self.left:
                self.right = False
            self.left = True

        if self.ordre == 'Gauche' and self.left:
            drone.move_right(100)  # Fait fait aller à gauche le drone
            if self.right:
                self.left = False
            self.right = True
        if self.ordre == 'Recule' and self.back:
            drone.move_back(100)  # Fait reculer le drone
            if self.forward:
                self.back = False
            self.forward = True

        if self.ordre == 'Avance' and self.forward:
            drone.move_forward(100)  # Fait avancer le drone
            if self.back:
                self.forward = False
            self.back = True