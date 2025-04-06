# Robot-Hand

**(Optional: Add a cool GIF or image of the project working here!)**
<!-- ![Demo GIF](./docs/demo.gif) -->

## Description

A Python project that uses OpenCV and MediaPipe to detect and track the joints of an index finger in real-time via a webcam. It calculates the bend angles of the MCP, PIP, and DIP joints and translates these angles into commands sent over serial to an Arduino, which controls three corresponding servo motors to mimic the finger's movement.

## Features

*   Real-time hand and finger landmark detection using MediaPipe
*   Calculation of index finger joint angles
*   Mapping of calculated human finger angles to servo motor ranges
*   Serial communication between Python script and Arduino
*   Control of 3 servo motors via Arduino based on detected finger angles

## Hardware Requirements

*   Computer with Python installed
*   Webcam
*   Arduino (e.g., Arduino Uno, Nano, ESP32)
*   3 x Servo Motors (Standard hobby servos like SG90 or MG996R are common)
*   Jumper Wires
*   Breadboard (Optional, but recommended)
*   USB Cable (for Arduino)
*   **Important:** External 5V Power Supply for Servos (Recommended, especially for >1-2 small servos, to avoid overloading the Arduino)

## Software Requirements & Dependencies

*   **Python 3.12**
*   **Python Libraries:**
    *   `opencv-python`
    *   `mediapipe`
    *   `numpy`
    *   `pyserial`
*   **Arduino IDE**

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/logandul/Robot-Hand.git
    cd Robot-Hand
    ```

2.  **Install Python Dependencies:**
    ```bash
    # pip install opencv-python mediapipe numpy pyserial
    ```

3.  **Hardware Connections:**
    *   Connect the servos to the Arduino:
        *   Servo GND -> Arduino GND
        *   Servo VCC -> External 5V Power Supply (+)
        *   Servo Signal -> Arduino PWM Pins (e.g., D9, D10, D11 - check the `.ino` file)
    *   **Crucially:** Connect the GND of the external power supply to the Arduino GND if using one.
    *   Connect the Arduino to your computer via USB.
    *   *(Optional: Add a simple wiring diagram image or Fritzing diagram link here)*

4.  **Upload Arduino Sketch:**
    *   Open the Arduino IDE.
    *   Open the `.ino` file provided in this repository (`Index_Finger_Servo_Arduino.ino`).
    *   Select the correct Board and Port under the `Tools` menu.
    *   Upload the sketch to your Arduino.

5.  **Configure Serial Port:**
    *   Identify the serial port your Arduino is connected to (e.g., `COM3` on Windows, `/dev/ttyACM0` or `/dev/cu.usbmodem...` on Linux/macOS). You can usually find this in the Arduino IDE (Tools -> Port).
    *   Open Index_Finger_Robot_Manipulation.py
    *   Find the `SERIAL_PORT` variable near the top and **change its value** to match your Arduino's port.

6.  **Calibration (VERY IMPORTANT):**
    *   Run the Python script initially just to observe the angle outputs.
    *   In the Python script 'Index_Finger_Robot_Manipulation.py', locate the **Servo Calibration Parameters** section.
    *   Adjust the `MIN/MAX_BEND_ANGLE_...` values based on the angles MediaPipe reports for your *actual* finger when it's straight vs. fully bent for each joint.
    *   Adjust the `SERVO_STRAIGHT_POS` and `SERVO_BENT_POS_...` values based on the physical range of motion of your servos and how they are mounted in your robotic finger/hand structure. **This requires experimentation!**

## Usage

1.  Ensure the Arduino is connected and the sketch is running.
2.  Make sure your webcam is connected.
3.  Run the main Python script from your terminal:
    ```bash
    python Index_Finger_Robot_Manipulation.py
    ```
4.  Position your hand in front of the webcam. The script should detect your index finger and the servos should start moving.
5.  Press 'q' in the OpenCV window to quit the application.

## How it Works (Brief Overview)

1.  **Capture:** OpenCV captures video frames from the webcam.
2.  **Detection:** MediaPipe processes the frame to detect hand landmarks in 3D space.
3.  **Angle Calculation:** Vector math (using NumPy) calculates the angles between the relevant finger bone segments (landmarks) for the MCP, PIP, and DIP joints.
4.  **Mapping:** The calculated human joint angles (e.g., 10°-90°) are mapped to the servo's operational range (e.g., 0°-180°) using linear interpolation based on the calibration parameters.
5.  **Communication:** The mapped servo angles are formatted into a command string (e.g., "90,45,30\n").
6.  **Transmission:** PySerial sends this command string over the USB serial connection to the Arduino.
7.  **Actuation:** The Arduino code parses the received string, extracts the individual angles, and uses the `Servo.write()` function to move each servo to the target position.

## Future Improvements / To-Do

*   [ ] Add control for other fingers.
*   [ ] Implement smoothing algorithms for less jittery servo movement.
*   [ ] Develop a simple GUI instead of just the OpenCV window.
*   [ ] Explore wireless communication (e.g., Bluetooth, ESP-NOW) instead of USB Serial.
*   [ ] Add gesture recognition to trigger specific actions.
*   [ ] Improve calibration process (e.g., interactive calibration).

## License
MIT License

## Acknowledgements

*   [MediaPipe](https://developers.google.com/mediapipe) by Google
*   [OpenCV](https://opencv.org/)
