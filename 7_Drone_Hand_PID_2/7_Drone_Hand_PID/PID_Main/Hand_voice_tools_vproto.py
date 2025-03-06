import numpy as np
import cv2 as cv
import math

def sinus(a):
    return np.sin(math.radians(a))
def cosinus(a):
    return np.cos(math.radians(a))

def Tracking_mask(im,ptn):
    # Position masque
    xmax=int(ptn[:,0].max())
    xmin=int(ptn[:,0].min())
    ymax=int(ptn[:,1].max())
    ymin=int(ptn[:,1].min())
    
    # Marge
    mx=int((xmax-xmin)*0.2)
    my=int((ymax-ymin)*0.1)
    
    # Masque
    mask=np.ones(im.shape)
    cv.rectangle(mask,(xmax+mx,ymin-my),(xmin-mx,ymax+my),(0,0,255),2)
    im*=mask.astype('uint8')
    
    return im
# def Tracking_mask(im,ptn):
#     # Position masque
#     xmax=int(ptn[:,0].max())
#     xmin=int(ptn[:,0].min())
#     ymax=int(ptn[:,1].max())
#     ymin=int(ptn[:,1].min())
    
#     # Marge
#     mx=int((xmax-xmin)*0.2)
#     my=int((ymax-ymin)*0.1)
    
#     # Masque
#     mask=np.zeros(im.shape)
#     cv.rectangle(mask,(xmax+mx,ymin-my),(xmin-mx,ymax+my),(1,1,1),-1)
#     im*=mask.astype('uint8')
    
#     return im

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

def Incl_compute(ptn,reference=9):
    
    Diff_ptnHB=ptn[reference,:]-ptn[0,:]
    hyp=np.sqrt(Diff_ptnHB[0]**2+Diff_ptnHB[1]**2)
    
    Angle=np.degrees(np.arccos(Diff_ptnHB[0]/hyp))*-(Diff_ptnHB[1]/abs(Diff_ptnHB[1]))
    return  Angle

def Barycentre(ptn):
    return ((ptn[0,:]+ptn[5,:]+ptn[17,:])/3).astype(int)
    
    
    
    
    
    
    
    
    
    
    
    