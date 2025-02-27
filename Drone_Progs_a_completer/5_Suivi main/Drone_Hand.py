import cv2
import mediapipe as mp
import numpy as np
import Fonction as fc

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def lm_extraction(results, annotated_img):
    lm = np.zeros((1, 21, 3))
    i = 0
    for hand_landmarks in results.multi_hand_landmarks:
        point = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        lm[i, 0, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
        lm[i, 1, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
        lm[i, 2, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        lm[i, 3, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        lm[i, 4, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        lm[i, 5, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        lm[i, 6, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
        lm[i, 7, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        lm[i, 8, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        lm[i, 9, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        lm[i, 10, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
        lm[i, 11, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        lm[i, 12, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        lm[i, 13, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        lm[i, 14, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
        lm[i, 15, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        lm[i, 16, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
        lm[i, 17, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        lm[i, 18, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
        lm[i, 19, :] = point.x, point.y, point.z

        point = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        lm[i, 20, :] = point.x, point.y, point.z

        i = i + 1

        # Acréation d'une image annoté des lm
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    return lm, annotated_image

def hands_mystere(img):
    S1=350
    S2=450
    c=[S1,S2,20,20]
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
        i=0
        # si il y a une image

        i_a = img.copy()
        cv2.namedWindow('video', cv2.WINDOW_NORMAL)
        image_height, image_width, __ = img.shape

        # recherche de main
        results = hands.process(i_a)
        # si il n'y a pas de main
        if not results.multi_hand_landmarks:
            if (i % 60) == 0:
                print('no hand detected')
            i += 1
        # sinon on extrait les données
        else:
            # extraction des lm
            lm, i_a = fc.lm_extraction(results, i_a)

            hand_xmin = int(np.amin(lm[0, :, 0]) * image_width)
            hand_xmax = int(np.amax(lm[0, :, 0]) * image_width)
            hand_ymin = int(np.amin(lm[0, :, 1]) * image_height)
            hand_ymax = int(np.amax(lm[0, :, 1]) * image_height)

            hand_centre = [(hand_xmax - hand_xmin) // 2, (hand_ymax - hand_ymin) // 2]
            c = [hand_xmin, hand_ymin, hand_xmax - hand_xmin, hand_ymax - hand_ymin]

            i_a = cv2.rectangle(i_a, c, (255, 0, 0), 5)

        return c,i_a