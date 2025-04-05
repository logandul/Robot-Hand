import cv2
import mediapipe as mp
import math
import numpy as np
import serial
import time

SERIAL_PORT = 'COM4'
BAUD_RATE = 9600
ser = None

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on {SERIAL_PORT}")
except serial.SerialException as e:
    print(f"Error connecting to {SERIAL_PORT}: {e}")
    print("Running without servo controls.")
    ser = None

def map_angle(value, from_min,from_max,to_min,to_max):
    value = np.clip(value,from_min,from_max)
    return np.interp(value,[from_min,from_max],[to_min,to_max])

MIN_BEND_ANGLE = 0
MAX_BEND_ANGLE = 180
SERVO_STRAIGHT_POS = 0
SERVO_BENT_POS = 120


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

                servo_MCP = map_angle(index_MCP_angle,MIN_BEND_ANGLE,MAX_BEND_ANGLE,SERVO_STRAIGHT_POS,SERVO_BENT_POS)
                servo_PIP = map_angle(index_PIP_angle,MIN_BEND_ANGLE,MAX_BEND_ANGLE,SERVO_STRAIGHT_POS,SERVO_BENT_POS)
                servo_DIP = map_angle(index_DIP_angle,MIN_BEND_ANGLE,MAX_BEND_ANGLE,SERVO_STRAIGHT_POS,SERVO_BENT_POS)



                servo_MCP_int = int(servo_MCP)
                servo_PIP_int = int(servo_PIP)
                servo_DIP_int = int(servo_DIP)

                print(servo_MCP_int)
                print(servo_PIP_int)
                print(servo_DIP_int)
                if ser and ser.is_open:
                    command = f"{servo_MCP_int},{servo_PIP_int},{servo_DIP_int}\n"
                    try:
                        ser.write(command.encode('utf-8'))
                    except serial.SerialExcpetion as e:
                        print(f"Error writing to serial port: {e}")
                        ser=None





    # Display the result
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
