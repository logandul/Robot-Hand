import cv2
import mediapipe as mp
import math
import numpy as np

def get_vector(p1,p2):
    return np.array([p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]])

def calc_angle(v1,v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle_rad)


# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Set up the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip for natural feel and convert BGR to RGB
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:


        for hand_landmarks in result.multi_hand_landmarks:


            # Draw landmarks and connections
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get hand landmark coordinates
            h, w, _ = frame.shape
            landmark_names = {
                0: "Wrist",
                1: "Thumb CMC",
                2: "Thumb MCP",
                3: "Thumb IP",
                4: "Thumb Tip",
                5: "Index MCP",
                6: "Index PIP",
                7: "Index DIP",
                8: "Index Tip",
                9: "Middle MCP",
                10: "Middle PIP",
                11: "Middle DIP",
                12: "Middle Tip",
                13: "Ring MCP",
                14: "Ring PIP",
                15: "Ring DIP",
                16: "Ring Tip",
                17: "Pinky MCP",
                18: "Pinky PIP",
                19: "Pinky DIP",
                20: "Pinky Tip"
            }

            h, w, _ = frame.shape
            for idx, lm in enumerate(hand_landmarks.landmark):
                # name = landmark_names.get(idx, f"Unknown-{idx}")
                # cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                # print(f"{name} ({idx}): x={cx}, y={cy}, z={cz:.4f}")

                coords = []

                for idx in range(21):
                    lm = hand_landmarks.landmark[idx]
                    coords.append([lm.x, lm.y, lm.z])


                # Define the angles between the index finger

                # PIP joint angles
                index_PIP_v1 = get_vector(coords[5], coords[6])
                index_PIP_v2 = get_vector(coords[7], coords[6])

                index_PIP_angle = calc_angle(index_PIP_v1, index_PIP_v2)
                print(f"Index PIP bend angle: {index_PIP_angle:.2f}°")

                # DIP joint angles
                index_DIP_v1 = get_vector(coords[6], coords[7])
                index_DIP_v2 = get_vector(coords[8], coords[7])

                index_DIP_angle = calc_angle(index_DIP_v1, index_DIP_v2)
                print(f"Index DIP bend angle: {index_DIP_angle:.2f}°")

                # MCP joint angles
                index_MCP_v1 = get_vector(coords[0], coords[5])
                index_MCP_v2 = get_vector(coords[6], coords[5])

                index_MCP_angle = calc_angle(index_MCP_v1, index_MCP_v2)
                print(f"Index MCP bend angle: {index_MCP_angle:.2f}°")

                # Define the angles between the thumb

                # PIP joint angles
                thumb_PIP_v1 = get_vector(coords[1], coords[2])
                thumb_PIP_v2 = get_vector(coords[3], coords[2])

                thumb_PIP_angle = calc_angle(thumb_PIP_v1, thumb_PIP_v2)
                print(f"Thumb PIP bend angle: {thumb_PIP_angle:.2f}°")

                # DIP joint angles
                thumb_DIP_v1 = get_vector(coords[2], coords[3])
                thumb_DIP_v2 = get_vector(coords[4], coords[3])

                thumb_DIP_angle = calc_angle(thumb_DIP_v1, thumb_DIP_v2)
                print(f"Thumb DIP bend angle: {thumb_DIP_angle:.2f}°")

                # MCP joint angles
                thumb_MCP_v1 = get_vector(coords[0], coords[1])
                thumb_MCP_v2 = get_vector(coords[2], coords[1])

                thumb_MCP_angle = calc_angle(thumb_MCP_v1, thumb_MCP_v2)
                print(f"Thumb MCP bend angle: {thumb_MCP_angle:.2f}°")

                # Define the angles between the middle finger

                # PIP joint angles
                middle_PIP_v1 = get_vector(coords[9], coords[10])
                middle_PIP_v2 = get_vector(coords[11], coords[10])

                middle_PIP_angle = calc_angle(middle_PIP_v1, middle_PIP_v2)
                print(f"Middle finger PIP bend angle: {middle_PIP_angle:.2f}°")

                # DIP joint angles
                middle_DIP_v1 = get_vector(coords[10], coords[11])
                middle_DIP_v2 = get_vector(coords[12], coords[11])

                middle_DIP_angle = calc_angle(middle_DIP_v1, middle_DIP_v2)
                print(f"Middle Finger DIP bend angle: {middle_DIP_angle:.2f}°")

                # MCP joint angles
                middle_MCP_v1 = get_vector(coords[0], coords[9])
                middle_MCP_v2 = get_vector(coords[10], coords[9])

                middle_MCP_angle = calc_angle(middle_MCP_v1, middle_MCP_v2)
                print(f"Middle Finger MCP bend angle: {middle_MCP_angle:.2f}°")

                # Define the angles between the ring finger

                # PIP joint angles
                ring_PIP_v1 = get_vector(coords[13], coords[14])
                ring_PIP_v2 = get_vector(coords[15], coords[14])

                ring_PIP_angle = calc_angle(ring_PIP_v1, ring_PIP_v2)
                print(f"Ring Finger PIP bend angle: {ring_PIP_angle:.2f}°")

                # DIP joint angles
                ring_DIP_v1 = get_vector(coords[14], coords[15])
                ring_DIP_v2 = get_vector(coords[16], coords[15])

                ring_DIP_angle = calc_angle(ring_DIP_v1, ring_DIP_v2)
                print(f"Ring Finger DIP bend angle: {ring_DIP_angle:.2f}°")

                # MCP joint angles
                ring_MCP_v1 = get_vector(coords[0], coords[13])
                ring_MCP_v2 = get_vector(coords[14], coords[13])

                ring_MCP_angle = calc_angle(ring_MCP_v1, ring_MCP_v2)
                print(f"Ringer Finger MCP bend angle: {ring_MCP_angle:.2f}°")

                # Define the angles between the pinkie finger

                # PIP joint angles
                pink_PIP_v1 = get_vector(coords[17], coords[18])
                pink_PIP_v2 = get_vector(coords[19], coords[18])

                pink_PIP_angle = calc_angle(pink_PIP_v1, pink_PIP_v2)
                print(f"Pinkie PIP bend angle: {pink_PIP_angle:.2f}°")

                # DIP joint angles
                pink_DIP_v1 = get_vector(coords[18], coords[19])
                pink_DIP_v2 = get_vector(coords[20], coords[19])

                pink_DIP_angle = calc_angle(pink_DIP_v1, pink_DIP_v2)
                print(f"Pinkie DIP bend angle: {pink_DIP_angle:.2f}°")

                # MCP joint angles
                pink_MCP_v1 = get_vector(coords[0], coords[17])
                pink_MCP_v2 = get_vector(coords[18], coords[17])

                pink_MCP_angle = calc_angle(pink_MCP_v1, pink_MCP_v2)
                print(f"Pinkie MCP bend angle: {pink_MCP_angle:.2f}°")



    # Display the result
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
