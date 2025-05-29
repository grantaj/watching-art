import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk

# Get actual screen size using tkinter
root = tk.Tk()
root.withdraw()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Set up camera
cap = cv2.VideoCapture(0)

# Smoothing parameters
smoothed_x = None
smoothed_y = None
alpha = 0.2  # adjust between 0 (no update) and 1 (no smoothing)

# Calibration state and data
calibrating = False
calibration_data = []
min_x, max_x = -0.5, 0.5
min_y, max_y = -0.5, 0.5


def normalized_to_pixel_coords(norm_x, norm_y, width, height):
    return np.array([norm_x * width, norm_y * height])


def get_3d_point(landmark, width, height):
    return np.array([landmark.x * width, landmark.y * height, landmark.z * width])


def compute_head_direction(nose, left_eye, right_eye):
    eye_vector = left_eye - right_eye  # flip direction for left-right correction
    nose_vector = nose - ((left_eye + right_eye) / 2)
    normal = np.cross(eye_vector, nose_vector)
    norm = np.linalg.norm(normal)
    return normal / (norm + 1e-6)


def update_calibration_bounds_live(new_normal):
    """
    Update min/max bounds in real-time for head_normal values (invert Y for calibration).
    """
    global min_x, max_x, min_y, max_y
    x_val = new_normal[0]
    y_val = -new_normal[1]  # invert Y for calibration
    min_x = min(min_x, x_val)
    max_x = max(max_x, x_val)
    min_y = min(min_y, y_val)
    max_y = max(max_y, y_val)


def update_calibration_bounds_from_history():
    """
    Compute calibration bounds from collected data using head_normal values.
    """
    global min_x, max_x, min_y, max_y
    if calibration_data:
        xs = [n[0] for n in calibration_data]
        ys = [n[1] for n in calibration_data]  # use raw head_normal y
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

def normalize_to_screen(x, y):
    """
    Map normalized head vector to screen coordinates based on calibration bounds.
    """
    x_val = x
    y_val = y  # raw head_normal y
    range_x = max_x - min_x if max_x != min_x else 1
    range_y = max_y - min_y if max_y != min_y else 1
    norm_x = (x_val - min_x) / range_x
    norm_y = (y_val - min_y) / range_y
    screen_x = int(norm_x * screen_width)
    # Positive head_normal y now maps downward
    screen_y = int(norm_y * screen_height)
    return screen_x, screen_y


def draw_landmarks(frame, landmarks, width, height):
    left_eye_px = normalized_to_pixel_coords(landmarks[33].x, landmarks[33].y, width, height)
    right_eye_px = normalized_to_pixel_coords(landmarks[263].x, landmarks[263].y, width, height)
    nose_tip_px = normalized_to_pixel_coords(landmarks[1].x, landmarks[1].y, width, height)
    center_eye_px = (left_eye_px + right_eye_px) / 2

    cv2.circle(frame, tuple(left_eye_px.astype(int)), 3, (255, 0, 0), -1)
    cv2.circle(frame, tuple(right_eye_px.astype(int)), 3, (0, 0, 255), -1)
    cv2.circle(frame, tuple(nose_tip_px.astype(int)), 3, (0, 255, 0), -1)
    cv2.line(frame, tuple(left_eye_px.astype(int)), tuple(right_eye_px.astype(int)), (255, 255, 0), 1)
    cv2.line(frame, tuple(center_eye_px.astype(int)), tuple(nose_tip_px.astype(int)), (255, 255, 255), 1)


def render_gaze_point(normal):
    global smoothed_x, smoothed_y
    # Live calibration updates
    if calibrating:
        update_calibration_bounds_live(normal)
        calibration_data.append(normal)

    # Map to screen coordinates
    x, y = normalize_to_screen(normal[0], normal[1])

    # Apply exponential smoothing
    if smoothed_x is None or smoothed_y is None:
        smoothed_x, smoothed_y = x, y
    else:
        smoothed_x = int((1 - alpha) * smoothed_x + alpha * x)
        smoothed_y = int((1 - alpha) * smoothed_y + alpha * y)

    # Create gaze frame
    gaze_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Draw calibration bounds for debugging
    top_bound = normalize_to_screen(0, min_y)[1]  # head_normal y=min => screen top
    bottom_bound = normalize_to_screen(0, max_y)[1]  # head_normal y=max => screen bottom
    cv2.line(gaze_frame, (0, top_bound), (screen_width, top_bound), (0, 0, 255), 2)
    cv2.line(gaze_frame, (0, bottom_bound), (screen_width, bottom_bound), (0, 0, 255), 2)

    # Draw smoothed gaze dot
    cv2.circle(gaze_frame, (smoothed_x, smoothed_y), 10, (0, 255, 0), -1)
    cv2.putText(gaze_frame, f"Gaze: ({smoothed_x}, {smoothed_y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    cv2.imshow('Gaze Point (Normal Vector)', gaze_frame)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        mesh = result.multi_face_landmarks[0].landmark
        nose_tip = get_3d_point(mesh[1], w, h)
        left_eye = get_3d_point(mesh[33], w, h)
        right_eye = get_3d_point(mesh[263], w, h)

        head_normal = compute_head_direction(nose_tip, left_eye, right_eye)
        draw_landmarks(frame, mesh, w, h)
        render_gaze_point(head_normal)

    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord('r') and not calibrating:
        calibrating = True
        calibration_data.clear()
        min_x, max_x = -0.5, 0.5
        min_y, max_y = -0.5, 0.5
        print("Calibration started. Move your head to the edges of the screen.")
    elif key == ord('s') and calibrating:
        calibrating = False
        update_calibration_bounds_from_history()
        print(f"Calibration stopped. Y-range: {min_y:.2f} to {max_y:.2f}")

cap.release()
cv2.destroyAllWindows()
