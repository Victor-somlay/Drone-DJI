cap = cv2.VideoCapture(0)

# Drone speed
speed = 25
# Distance from screen reference
distance_ref = 60  # cm
# Speed limit
max_speed = 40

# Number of detections
n_detection = 0
n_total_frames = 0

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:  # initial confidences = 0.5

    while cap.isOpened():
        n_total_frames += 1

        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
