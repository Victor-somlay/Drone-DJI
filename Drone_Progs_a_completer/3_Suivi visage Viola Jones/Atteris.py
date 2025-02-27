from djitellopy import Tello

# init Tello
tello = Tello()
# Tello setup
tello.connect()

tello.land()
tello.end()