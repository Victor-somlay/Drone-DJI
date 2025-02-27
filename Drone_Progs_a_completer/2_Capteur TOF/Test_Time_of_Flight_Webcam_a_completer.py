import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np

from time import sleep

from djitellopy import TelloSwarm

from time import sleep

from djitellopy import Tello
import math

tello1 = Tello()

tello1.connect()

# tello1.takeoff()

# 1) Affichage de l'accélération et d'autres paramètres de vol
print(tello1.get_acceleration_x())
# Affichez quelques informations comme
# la vitesse selon z, l'accélération selon z, la hauteur, etc.



KNOWN_DISTANCE = 76.2  # centimeter
# width of face in the real world or Object Plane
KNOWN_WIDTH = 14.3  # centimeter
# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX


# cap = cv2.VideoCapture(1)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y_f = lfilter(b, a, data)
    return y_f


# Setting standard filter requirements.
order = 2  # 2
fs = 30.0
cutoff = 2  # 3.667

b, a = butter_lowpass(cutoff, fs, order)

# face detector object
# face_detector = cv2.CascadeClassifier("drone_test_2\good\drone_2_OK_2.xml")
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
show_animation = True


def face_data(image):
    face_width = 0
    vecteur_y = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_shape = image.shape
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cv2.circle(frame, (x + w // 2, y + h // 2), 5, (0, 0, 255))
        face_width = h

        face_middle = (x + w // 2, y + h // 2)  # middle of face coordinate
        frame_middle = (frame_shape[1] // 2, frame_shape[0] // 2)  # middle of frame coordinate
        vecteur_y = np.array(frame_middle[0] - face_middle[0])
        # sleep(0.05)
        # print(vecteur_y )

    return face_width, vecteur_y


tt = [0.0]
x11a = [0.0]
y11a = [0.0]
Distance = 0
Distance_Y = 0
x11a_2 = [0.0]
A = [0.0]
A1 = 0
A1_2 = [0.0]
A2 = 0

A_af = [0.0]
A2_2 = [0.0]
A3 = 3
f = True
zz1 = [0.0]


def Affiche(t, x11af, y11af, zzaf):
    # A = 0
    # print(t,x11af,y11af )

    tt.append(t)
    x11a.append(x11af)  # x11af

    y11a.append(y11af)
    zz1.append(zzaf)
    # print(zz1)

    X_f = butter_lowpass_filter(x11a, cutoff, fs, order)
    Y_f = butter_lowpass_filter(y11a, cutoff, fs, order)

    print(t, x11a[-1], X_f[-1], y11a[-1], Y_f[-1])

    if show_animation:
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 's' else None])

        plt.plot(tt, x11a, "-g", label="trajectory")
        # plt.plot(tt,X_f, "-r")
        plt.plot(tt, y11a, "-r")
        # plt.plot(tt, Y_f, "-b")
        # plt.plot(x11a, y11a, "b--")
        # plt.plot(X_f, Y_f, "r")
        # fig = plt.figure()
        # Distance_To_Ground_ToF = plt.Distance_To_Ground_ToFes(projection='3d')
        # Distance_To_Ground_ToF.plot3D(x11a, y11a, x11a, 'gray')

        # plt.xlim([-300,  300])
        # plt.ylim([-100,  100])
        # plt.plot(tt, tt, "-r", label="trajectory")
        # plt.grid(True)

        plt.pause(0.0001)  # 0.0001
    return


Distance_To_Ground_ToF = 0

# reading reference image from directory
# ref_image = cv2.imread("000001.jpg")

# ref_image_face_width = face_data(ref_image)
# print(ref_image_face_width)
focal_length_found = 970.0  # focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_face_width)
# print(KNOWN_WIDTH * focal_length_found)
# cv2.imshow("ref_image", ref_image)


start = time.time()
cap = cv2.VideoCapture(0)

tttt = [0.0]
dtt = 0
dt = 0.05
v_x = 0
last_Distance = 0

error_x = 0
speed_x = 0
pError_x = 0
last_error_x = 0

while True:

    end = time.time()
    t = end - start
    _, frame = cap.read()

    # calling face_data function
    face_width_in_frame = face_data(frame)
    # finding the distance by calling function Distance
    if face_width_in_frame[0] != 0:
        # Distance =  distance_finder(focal_length_found, KNOWN_WIDTH, face_width_in_frame)

        # Distance = int(13871.0/ (face_width_in_frame[0]) )#(KNOWN_WIDTH * focal_length_found) / face_width_in_frame
        Distance = (int(10000 / face_width_in_frame[0]))

        Distance_Y = int(face_width_in_frame[1] * (Distance / 700))

        # Drwaing Text on the screen
        cv2.putText(
            frame, f"Distance_x = {round(Distance, 2)} CM", (50, 50), fonts, 1, (WHITE), 2
        )

        cv2.putText(
            frame, f"Distance_y = {round(Distance_Y, 2)} CM", (50, 100), fonts, 1, (WHITE), 2
        )

    Distance_To_Ground_ToF = tello1.get_distance_tof()

    Acceleration_z = tello1.get_acceleration_z()
    tttt.append(t)
    dtt = tttt[-2]
    # dt = t -dt
    dt = t - dtt

# 2) Calcul de la consigne de distance au Sol
    error_x =  # Distance de consigne - distance mesurée (en cm)

    pError_x +=   # terme proportionnel
    dError_x =    # terme dérivée
    last_error_x = error_x

# 4) Calcul de la commande:
    speed = 0.8 * erreur + 0.01 * erreur_derivee  + 0.0001* erreur_integree

# 5) restriction de la valeur de commande à l'intervalle -20 : 20:
    speed = # avec 'int' et 'clip'

    # print(error_x,pError_x, Distance, speed)
    # sleep(0.2)
    # v_x = (Distance - last_Distance) / dt
    # last_Distance = Distance
    # Distance += v_x * dt

#3) programmer un mouvement vertical qui prend en compte le paramètre speed.
    tello1.???



    Affiche(t, Distance_To_Ground_ToF, -Acceleration_z / 10, 0)
    # Affiche(t, Distance, Distance_Y, 0)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        tello1.land()
        break
tello1.land()
cap.release()
cv2.destroyAllWindows()
