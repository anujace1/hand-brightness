import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.75,
    min_hand_presence_confidence=0.75,
    min_tracking_confidence=0.75
)

hand_landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

frame_timestamp = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Convert OpenCV BGR -> RGB
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frameRGB
    )

    # Detect hands
    frame_timestamp += 1

    result = hand_landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )

    landmarkList = []

    height, width, _ = frame.shape

    # Process detected hands
    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            for _id, landmark in enumerate(hand_landmarks):

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                landmarkList.append([_id, x, y])

                # Draw landmark 
                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    cv2.FILLED
                )

            # Draw connections manually
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]

            for start, end in connections:

                x1 = int(hand_landmarks[start].x * width)
                y1 = int(hand_landmarks[start].y * height)

                x2 = int(hand_landmarks[end].x * width)
                y2 = int(hand_landmarks[end].y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

    # Use thumb tip (4) and index finger tip (8)
    if len(landmarkList) >= 21:

        x_1, y_1 = landmarkList[4][1], landmarkList[4][2]
        x_2, y_2 = landmarkList[8][1], landmarkList[8][2]

        cv2.circle(
            frame,
            (x_1, y_1),
            7,
            (0, 255, 0),
            cv2.FILLED
        )

        cv2.circle(
            frame,
            (x_2, y_2),
            7,
            (0, 255, 0),
            cv2.FILLED
        )

        cv2.line(
            frame,
            (x_1, y_1),
            (x_2, y_2),
            (0, 255, 0),
            3
        )

        # Distance between thumb and index finger
        L = hypot(
            x_2 - x_1,
            y_2 - y_1
        )

        # Convert distance to brightness
        b_level = np.interp(
            L,
            [15, 220],
            [0, 100]
        )

        sbc.set_brightness(int(b_level))

    cv2.imshow("Screen", frame)

    if cv2.waitKey(1) & 0xff == ord("s"):
        break

cap.release()
sbc.set_brightness(90)
hand_landmarker.close()
cv2.destroyAllWindows()