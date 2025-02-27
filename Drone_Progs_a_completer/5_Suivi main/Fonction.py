#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system('pip install mediapipe')


# In[3]:


import numpy as np
import cv2 as cv
import mediapipe as mp
import os

mp_hands = mp.solutions.hands

# In[ ]:


def lm_distance(lm):
  dist=np.zeros((2,21,3))

  # ditance de la paume
  p1 = 9
  p2 = 0
  dist[0, 0, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 0, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])

  #pouce  (p1=4  p2=2)
  p1=4
  p2=2
  dist[0,1, 0]=np.linalg.norm(lm[0,p1,:]-lm[0,p2,:])
  dist[1, 1, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])

  # index  (p1=8  p2=5)
  p1 = 8
  p2 = 5
  dist[0, 2, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 2, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])

  # mùajeur  (p1=12  p2=9)
  p1 = 12
  p2 = 9
  dist[0, 3, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 3, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])

  # annulaire  (p1=16  p2=13)
  p1 = 16
  p2 = 13
  dist[0, 4, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 4, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])

  # auriculaire  (p1=20  p2=17)
  p1 = 20
  p2 = 17
  dist[0, 5, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 5, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])


  #DISTANCE ENTRE LES DOIGTS
  #liens au pouce (p1=4)
  p1=4
  p2=8
  dist[0, 6, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 6, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])
  p2 = 12
  dist[0, 7, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 7, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])
  p2 = 16
  dist[0, 8, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 8, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])
  p2 = 20
  dist[0, 9, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 9, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])
  p2 = 0
  dist[0, 10, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 10, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])

  # liens a l'index (p1=8)
  p1=8
  p2 = 12
  dist[0, 11, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 11, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])
  p2 = 16
  dist[0, 12, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 12, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])
  p2 = 20
  dist[0, 13, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 13, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])
  p2 = 0
  dist[0, 14, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 14, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])

  # liens au majeur (p1=12)
  p1=12
  p2 = 16
  dist[0, 15, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 15, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])
  p2 = 20
  dist[0, 16, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 16, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])
  p2 = 0
  dist[0, 17, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 17, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])

  # liens a l'anulaire (p1=16)
  p1 = 16
  p2 = 20
  dist[0, 18, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 18, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])
  p2 = 0
  dist[0, 19, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 19, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])

  # liens a l'oriculair (p1=20)
  p1 = 20
  p2 = 0
  dist[0, 20, 0] = np.linalg.norm(lm[0, p1, :] - lm[0, p2, :])
  dist[1, 20, 0] = np.linalg.norm(lm[1, p1, :] - lm[1, p2, :])

  return dist


# In[ ]:


def lm_extraction (results,annotated_img):

  lm=np.zeros((2,21,3))
  i=0
  for hand_landmarks in results.multi_hand_landmarks:

    point=hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    lm[i,0,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    lm[i,1,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    lm[i,2,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    lm[i,3,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    lm[i,4,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    lm[i,5,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    lm[i,6,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    lm[i,7,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    lm[i,8,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    lm[i,9,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    lm[i,10,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    lm[i,11,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    lm[i,12,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    lm[i,13,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    lm[i,14,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    lm[i,15,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    lm[i,16,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    lm[i,17,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    lm[i,18,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    lm[i,19,:]=point.x,point.y,point.z

    point=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    lm[i,20,:]=point.x,point.y,point.z

    i=i+1
    '''
    #Acréation d'une image annoté des lm
    mp_drawing.draw_landmarks(
          annotated_img,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    
    '''

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    mp_drawing.draw_landmarks(
      annotated_img,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style()
    )

  dist=lm_distance(lm)
  lm_dist=np.zeros((2,21,4))
  lm_dist[:,:,0:3]=lm
  lm_dist[:, :,3] = dist[:,:,0]
  return lm_dist,annotated_img


# In[ ]:


'''
def save_lm(lm,cl):
  import csv
  chemin = "PFE.csv"
  with open(chemin, mode='w') as mon_fichier:
    mon_fichier_ecrire = csv.writer(mon_fichier, delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
    mon_fichier_ecrire.writerow(lm)
    print("{} sav in D:\Documents\A Ziane\A MASTER 2\PFE.csv".format(cl))
  return
'''


def save_lm(lm, folder):

  if not os.path.exists(folder):
    os.makedirs(folder)
  file_list = [f for f in os.listdir(folder) if f.startswith("lm_") and f.endswith(".npy")]
  file_count = len(file_list)
  filename = folder + "/lm_" + str(file_count) + ".npy"
  np.save(filename, lm)
  return 0


# In[ ]:


'''
# Utilisation de la fonction
save_lm(lm, "./data")

Cela enregistrera votre variable lm dans le dossier ./data 
avec un nom de fichier en utilisant un compteur qui incrémente à chaque appel de la fonction save_lm. 
Ainsi, le premier appel enregistrera un fichier nommé lm_0.npy, le deuxième appel enregistrera un fichier nommé lm_1.npy,
et ainsi de suite.

'''



def load_lm(folder):
  file_list = [f for f in os.listdir(folder) if f.startswith("lm_") and f.endswith(".npy")]
  file_list.sort()
  data = [np.load(folder + "/" + f) for f in file_list]
  return data

