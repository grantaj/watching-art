import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 3D model points for pose estimation (nose, eyes, mouth)
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -330.0, -65.0),   # Chin
    (-225.0, 170.0, -135.0), # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

while True:
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        h, w, _ = image.shape
        landmarks = results.multi_face_landmarks[0].landmark

        # Get 2D image points
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),    # Nose tip
            (landmarks[152].x * w, landmarks[152].y * h), # Chin
            (landmarks[33].x * w, landmarks[33].y * h),   # Left eye left
            (landmarks[263].x * w, landmarks[263].y * h), # Right eye right
            (landmarks[78].x * w, landmarks[78].y * h),   # Left mouth corner
            (landmarks[308].x * w, landmarks[308].y * h)  # Right mouth corner
        ], dtype="double")

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)

        # Draw nose direction
        nose_end = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]),
            rotation_vector, translation_vector, camera_matrix, dist_coeffs)[0]

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end[0][0][0]), int(nose_end[0][0][1]))
        cv2.line(image, p1, p2, (255, 0, 0), 2)

    cv2.imshow('Gaze Tracker', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
