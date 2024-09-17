import cv2
import mediapipe as mp
import numpy as np

class HandPoseRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hand landmarks
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Get hand gesture (you can implement your own gesture recognition logic here)
                gesture = self.recognize_gesture(hand_landmarks)

                # Display the recognized gesture
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def recognize_gesture(self, landmarks):
        # Implement your gesture recognition logic here
        # This is a placeholder function
        return "Unknown"

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Process the frame
            processed_frame = self.process_frame(frame)

            # Display the resulting frame
            cv2.imshow('Hand Pose Recognition', processed_frame)

            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_pose_recognition = HandPoseRecognition()
    hand_pose_recognition.run()
