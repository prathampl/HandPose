from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)

class HandPoseRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hand landmarks
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand gesture (you can implement your own gesture recognition logic here)
                gesture = self.recognize_gesture(hand_landmarks)
                return gesture

        return "No hand detected"

    def recognize_gesture(self, landmarks):
        # Implement your gesture recognition logic here
        # Placeholder for actual gesture recognition
        return "Unknown Gesture"

# Initialize the hand pose recognition model
hand_pose_recognition = HandPoseRecognition()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    img = Image.open(BytesIO(file.read()))
    img = np.array(img)

    # Convert RGBA to RGB if needed
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Process the image to recognize the hand gesture
    gesture = hand_pose_recognition.process_frame(img)

    # Return the gesture as JSON
    return jsonify({"gesture": gesture})

if __name__ == '__main__':
    app.run(debug=True)
