from djitellopy import Tello

# init Tello
tello = Tello()
# Tello setup
tello.connect()

tello.rotate_counter_clockwise(5)

