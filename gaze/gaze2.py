import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)

# Landmark indices
LEFT_EYE_INDICES = [33, 133, 159, 145]  # outer, inner, top, bottom
RIGHT_EYE_INDICES = [362, 263, 386, 374]
LEFT_IRIS_INDICES = [468, 469, 470, 471]  # 4 points on iris, center is approx their average
RIGHT_IRIS_INDICES = [473, 474, 475, 476]

def average_point(points):
    return np.mean(points, axis=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark

        def get_coords(indices):
            return np.array([[mesh[i].x * w, mesh[i].y * h] for i in indices], dtype=np.float32)

        # Left Eye
        left_eye = get_coords(LEFT_EYE_INDICES)
        left_iris = get_coords(LEFT_IRIS_INDICES)
        left_iris_center = average_point(left_iris)

        # Right Eye
        right_eye = get_coords(RIGHT_EYE_INDICES)
        right_iris = get_coords(RIGHT_IRIS_INDICES)
        right_iris_center = average_point(right_iris)

        # Draw left eye box
        cv2.polylines(frame, [left_eye.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.circle(frame, tuple(left_iris_center.astype(int)), 2, (0, 255, 255), -1)

        # Draw right eye box
        cv2.polylines(frame, [right_eye.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.circle(frame, tuple(right_iris_center.astype(int)), 2, (0, 255, 255), -1)

        # Estimate eye directions as vector from eye center to iris center
        left_eye_center = average_point([left_eye[0], left_eye[1]])  # outer + inner
        right_eye_center = average_point([right_eye[0], right_eye[1]])

        left_gaze_vector = left_iris_center - left_eye_center
        right_gaze_vector = right_iris_center - right_eye_center

        # Normalize for display
        def draw_gaze_vector(origin, vector, color):
            norm_vector = vector / (np.linalg.norm(vector) + 1e-6)
            endpoint = origin + norm_vector * 40  # scale for visibility
            cv2.line(frame, tuple(origin.astype(int)), tuple(endpoint.astype(int)), color, 2)

        draw_gaze_vector(left_eye_center, left_gaze_vector, (255, 0, 0))
        draw_gaze_vector(right_eye_center, right_gaze_vector, (255, 0, 0))

    cv2.imshow("Eye Direction", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
