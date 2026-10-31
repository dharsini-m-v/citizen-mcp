import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading
import pygame
import mysql.connector
from datetime import datetime
from ultralytics import YOLO


# ==========================================================
# CONFIGURATION
# ==========================================================

CAMERA_INDEX = 0

EYE_CLOSED_THRESHOLD = 0.21
EYE_CLOSED_DURATION = 2.0

YAWN_THRESHOLD = 0.60
YAWN_DURATION = 1.5

DISTRACTION_DURATION = 2.0
ALERT_COOLDOWN = 3.0

LOG_TO_MYSQL = False

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": "driver_monitoring"
}


# ==========================================================
# INITIALIZE MEDIAPIPE
# ==========================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================================
# INITIALIZE YOLO
# ==========================================================

# Downloads the pretrained YOLO model on first use if needed.
model = YOLO("yolov8n.pt")


# ==========================================================
# AUDIO ALERT SYSTEM
# ==========================================================

pygame.mixer.init()

ALERT_FILE = os.path.join("alerts", "alert.wav")


def play_alert():
    """Play a warning sound if the sound file exists."""

    if os.path.exists(ALERT_FILE):
        try:
            pygame.mixer.music.load(ALERT_FILE)
            pygame.mixer.music.play()
        except pygame.error:
            print("Unable to play alert sound.")
    else:
        print("\a", end="", flush=True)


# ==========================================================
# MYSQL DATABASE
# ==========================================================

db_connection = None
db_cursor = None


def initialize_database():
    """Connect to MySQL and create the event table."""

    global db_connection, db_cursor

    if not LOG_TO_MYSQL:
        print("MySQL logging disabled.")
        return

    try:
        db_connection = mysql.connector.connect(**MYSQL_CONFIG)
        db_cursor = db_connection.cursor()

        db_cursor.execute("""
            CREATE TABLE IF NOT EXISTS driver_events (
                id INT AUTO_INCREMENT PRIMARY KEY,
                event_type VARCHAR(100) NOT NULL,
                event_time DATETIME NOT NULL
            )
        """)

        db_connection.commit()
        print("MySQL connected successfully.")

    except mysql.connector.Error as error:
        print("MySQL connection failed:", error)
        db_connection = None
        db_cursor = None


def log_event(event_type):
    """Save a detected event to MySQL."""

    if db_connection is None or db_cursor is None:
        return

    try:
        db_cursor.execute(
            """
            INSERT INTO driver_events (event_type, event_time)
            VALUES (%s, %s)
            """,
            (event_type, datetime.now())
        )

        db_connection.commit()

    except mysql.connector.Error as error:
        print("Database logging error:", error)


# ==========================================================
# FACE LANDMARK UTILITIES
# ==========================================================

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""

    return np.linalg.norm(np.array(point1) - np.array(point2))


def eye_aspect_ratio(landmarks, eye_indices, width, height):
    """Estimate eye opening using facial landmark distances."""

    points = []

    for index in eye_indices:
        landmark = landmarks[index]
        points.append((
            int(landmark.x * width),
            int(landmark.y * height)
        ))

    vertical_1 = calculate_distance(points[1], points[5])
    vertical_2 = calculate_distance(points[2], points[4])
    horizontal = calculate_distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, width, height):
    """Estimate mouth opening using facial landmarks."""

    upper = landmarks[13]
    lower = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]

    upper_point = (upper.x * width, upper.y * height)
    lower_point = (lower.x * width, lower.y * height)
    left_point = (left.x * width, left.y * height)
    right_point = (right.x * width, right.y * height)

    vertical = calculate_distance(upper_point, lower_point)
    horizontal = calculate_distance(left_point, right_point)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


# MediaPipe Face Mesh landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ==========================================================
# DRIVER MONITORING
# ==========================================================

def run_driver_monitor():

    initialize_database()

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    eye_closed_start = None
    yawn_start = None
    distraction_start = None

    last_alert_time = {}
    previous_events = set()

    print("Driver Monitoring System Started.")
    print("Press Q to quit.")

    try:
        while True:

            success, frame = camera.read()

            if not success:
                print("Unable to read camera frame.")
                break

            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            current_events = set()

            # --------------------------------------------------
            # FACE MONITORING
            # --------------------------------------------------

            if results.multi_face_landmarks:

                face = results.multi_face_landmarks[0]
                landmarks = face.landmark

                left_ear = eye_aspect_ratio(
                    landmarks, LEFT_EYE, width, height
                )

                right_ear = eye_aspect_ratio(
                    landmarks, RIGHT_EYE, width, height
                )

                average_ear = (left_ear + right_ear) / 2

                mouth_ratio = mouth_aspect_ratio(
                    landmarks, width, height
                )

                # ----------------------------------------------
                # DROWSINESS DETECTION
                # ----------------------------------------------

                if average_ear < EYE_CLOSED_THRESHOLD:

                    if eye_closed_start is None:
                        eye_closed_start = time.time()

                    elif time.time() - eye_closed_start >= EYE_CLOSED_DURATION:
                        current_events.add("DROWSINESS")

                else:
                    eye_closed_start = None

                # ----------------------------------------------
                # YAWNING DETECTION
                # ----------------------------------------------

                if mouth_ratio > YAWN_THRESHOLD:

                    if yawn_start is None:
                        yawn_start = time.time()

                    elif time.time() - yawn_start >= YAWN_DURATION:
                        current_events.add("YAWNING")

                else:
                    yawn_start = None

                # ----------------------------------------------
                # BASIC HEAD-POSE / DISTRACTION ESTIMATE
                # ----------------------------------------------

                nose = landmarks[1]
                left_face = landmarks[234]
                right_face = landmarks[454]

                face_center = (left_face.x + right_face.x) / 2

                # Approximate horizontal head orientation.
                # This is a heuristic, not calibrated head-pose estimation.
                head_offset = abs(nose.x - face_center)

                if head_offset > 0.12:

                    if distraction_start is None:
                        distraction_start = time.time()

                    elif (
                        time.time() - distraction_start
                        >= DISTRACTION_DURATION
                    ):
                        current_events.add("DISTRACTION")

                else:
                    distraction_start = None

                # Display facial measurements.
                cv2.putText(
                    frame,
                    f"Eye Ratio: {average_ear:.2f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Mouth Ratio: {mouth_ratio:.2f}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

            else:
                eye_closed_start = None
                yawn_start = None
                distraction_start = None

            # --------------------------------------------------
            # MOBILE PHONE DETECTION USING YOLO
            # --------------------------------------------------

            predictions = model.predict(
                frame,
                conf=0.40,
                verbose=False
            )

            for prediction in predictions:

                for box in prediction.boxes:

                    class_id = int(box.cls[0])

                    class_name = model.names[class_id]

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    # COCO class 67 represents a cell phone.
                    if class_name == "cell phone":

                        current_events.add("MOBILE_PHONE")

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            "Mobile Phone",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )

            # --------------------------------------------------
            # ALERTS AND EVENT LOGGING
            # --------------------------------------------------

            for event in current_events:

                last_time = last_alert_time.get(event, 0)

                if time.time() - last_time >= ALERT_COOLDOWN:

                    print(
                        f"[ALERT] {event} detected at "
                        f"{datetime.now().strftime('%H:%M:%S')}"
                    )

                    threading.Thread(
                        target=play_alert,
                        daemon=True
                    ).start()

                    log_event(event)
                    last_alert_time[event] = time.time()

            # --------------------------------------------------
            # DISPLAY STATUS
            # --------------------------------------------------

            if current_events:

                status = "WARNING: " + ", ".join(
                    sorted(current_events)
                )

                color = (0, 0, 255)

            else:

                status = "Driver Status: NORMAL"
                color = (0, 255, 0)

            cv2.rectangle(
                frame,
                (0, height - 45),
                (width, height),
                (30, 30, 30),
                -1
            )

            cv2.putText(
                frame,
                status,
                (15, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            cv2.imshow(
                "AI-Based Driver Monitoring System",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()

        face_mesh.close()

        if db_cursor is not None:
            db_cursor.close()

        if db_connection is not None and db_connection.is_connected():
            db_connection.close()

        pygame.mixer.quit()

        print("Driver Monitoring System Stopped.")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    run_driver_monitor()
