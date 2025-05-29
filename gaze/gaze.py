import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
iris_indices = {
    "left": [474, 475, 476, 477],   # Iris landmarks (left eye)
    "right": [469, 470, 471, 472],  # Iris landmarks (right eye)
    "left_eye": [33, 133],          # Eye corners (left)
    "right_eye": [362, 263],        # Eye corners (right)
}

# Set up camera
cap = cv2.VideoCapture(0)
screen_width, screen_height = 640, 480  # For display window

# Initialize smoothing variables
smoothed_x = None
smoothed_y = None
alpha = 0.2  # smoothing factor (between 0 and 1)

def normalized_to_pixel_coords(norm_x, norm_y, width, height):
    return int(norm_x * width), int(norm_y * height)

def get_iris_center(landmarks, indices, w, h):
    pts = [normalized_to_pixel_coords(landmarks[i].x, landmarks[i].y, w, h) for i in indices]
    center = np.mean(pts, axis=0).astype(int)
    return tuple(center)

def compute_gaze_ratio(iris_center, corner1, corner2):
    if corner1[0] > corner2[0]:
        corner1, corner2 = corner2, corner1
    eye_width = max(1.0, corner2[0] - corner1[0])
    iris_x = np.clip(iris_center[0], corner1[0], corner2[0])
    gaze_ratio = (iris_x - corner1[0]) / eye_width
    return gaze_ratio

def compute_vertical_ratio(iris_center, corner1, corner2):
    if corner1[1] > corner2[1]:
        corner1, corner2 = corner2, corner1
    eye_height = max(1.0, abs(corner2[1] - corner1[1]))
    iris_y = np.clip(iris_center[1], corner1[1], corner2[1])
    gaze_ratio = 1.0 - (iris_y - corner1[1]) / eye_height
    return gaze_ratio

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror view
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        mesh = result.multi_face_landmarks[0].landmark

        # Get all relevant landmarks
        left_iris = get_iris_center(mesh, iris_indices["left"], w, h)
        right_iris = get_iris_center(mesh, iris_indices["right"], w, h)

        left_eye_corner1 = normalized_to_pixel_coords(mesh[iris_indices["left_eye"][0]].x, mesh[iris_indices["left_eye"][0]].y, w, h)
        left_eye_corner2 = normalized_to_pixel_coords(mesh[iris_indices["left_eye"][1]].x, mesh[iris_indices["left_eye"][1]].y, w, h)

        right_eye_corner1 = normalized_to_pixel_coords(mesh[iris_indices["right_eye"][0]].x, mesh[iris_indices["right_eye"][0]].y, w, h)
        right_eye_corner2 = normalized_to_pixel_coords(mesh[iris_indices["right_eye"][1]].x, mesh[iris_indices["right_eye"][1]].y, w, h)

        # Compute gaze ratios with corrected associations
        left_ratio = compute_gaze_ratio(left_iris, right_eye_corner1, right_eye_corner2)
        right_ratio = compute_gaze_ratio(right_iris, left_eye_corner1, left_eye_corner2)
        average_ratio_x = (left_ratio + right_ratio) / 2

        # Compute vertical ratios from left iris and right eye corners
        vertical_ratio = compute_vertical_ratio(left_iris, right_eye_corner1, right_eye_corner2)

        # Map to screen space
        target_x = int(average_ratio_x * screen_width)
        target_y = int(vertical_ratio * screen_height)

        # Apply exponential smoothing
        if smoothed_x is None:
            smoothed_x = target_x
            smoothed_y = target_y
        else:
            smoothed_x = int((1 - alpha) * smoothed_x + alpha * target_x)
            smoothed_y = int((1 - alpha) * smoothed_y + alpha * target_y)

        # Draw iris centers and ratios
        cv2.circle(frame, left_iris, 3, (0, 255, 0), -1)
        cv2.putText(frame, f"left iris ratio: {left_ratio:.2f}", (left_iris[0]+5, left_iris[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.circle(frame, right_iris, 3, (0, 255, 255), -1)
        cv2.putText(frame, f"right iris ratio: {right_ratio:.2f}", (right_iris[0]+5, right_iris[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # Draw average gaze point on a separate frame
        gaze_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        cv2.circle(gaze_frame, (smoothed_x, smoothed_y), 10, (0, 255, 0), -1)
        cv2.putText(gaze_frame, f"x: {average_ratio_x:.2f} y: {vertical_ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        cv2.imshow('Gaze Point', gaze_frame)

        # Draw eye corners
        cv2.circle(frame, left_eye_corner1, 2, (255, 0, 0), -1)
        cv2.circle(frame, left_eye_corner2, 2, (255, 0, 0), -1)
        cv2.putText(frame, "left corners", (left_eye_corner1[0], left_eye_corner1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

        cv2.circle(frame, right_eye_corner1, 2, (0, 0, 255), -1)
        cv2.circle(frame, right_eye_corner2, 2, (0, 0, 255), -1)
        cv2.putText(frame, "right corners", (right_eye_corner1[0], right_eye_corner1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

        # Association lines
        cv2.line(frame, left_iris, right_eye_corner1, (0, 255, 0), 1)
        cv2.line(frame, left_iris, right_eye_corner2, (0, 255, 0), 1)
        cv2.line(frame, right_iris, left_eye_corner1, (0, 255, 255), 1)
        cv2.line(frame, right_iris, left_eye_corner2, (0, 255, 255), 1)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
