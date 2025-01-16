import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
from PIL import Image
import tempfile
import os

# Load your trained LSTM model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")
st.success("Model loaded successfully!")

# Define actions corresponding to the output classes of the model
actions = ["A", "B", "C"]  # Replace with your model's actions

# Generate 20 colors for visualization
colors = [(245, 117, 16)] * len(actions)

# Helper function to visualize probabilities
def prob_viz(res, actions, input_frame, threshold=0.8):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        x1, y1 = 0, 60 + num * 40
        x2, y2 = int(prob * 100), 90 + num * 40
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {int(prob * 100)}%",
                    (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output_frame

# Function for mediapipe detection and keypoint extraction
def mediapipe_detection(image, hands):
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        keypoints = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])
        return np.array(keypoints).flatten()
    return np.zeros(21 * 3)

# Initialize Streamlit app
st.title("Sign Language Recognition")

# Session state for sequence storage
if "sequence" not in st.session_state:
    st.session_state["sequence"] = []
if "sentence" not in st.session_state:
    st.session_state["sentence"] = []
if "accuracy" not in st.session_state:
    st.session_state["accuracy"] = []

# Start camera button
start_camera = st.button("Start Camera")
if start_camera:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to access the camera. Ensure it's connected and not in use.")

# Capture frame and detect sign button
capture_and_detect = st.button("Capture and Detect Sign")
if capture_and_detect:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Crop the frame and draw the ROI
            crop_frame = frame[40:400, 0:300]
            cv2.rectangle(frame, (0, 40), (300, 400), (255, 0, 0), 2)

            # Load MediaPipe Hands model
            import mediapipe as mp
            mp_hands = mp.solutions.hands
            with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                # Detect hand landmarks
                image, results = mediapipe_detection(crop_frame, hands)

                # Extract keypoints
                keypoints = extract_keypoints(results)
                st.session_state["sequence"].append(keypoints)
                st.session_state["sequence"] = st.session_state["sequence"][-30:]  # Last 30 frames

                # Perform prediction if enough frames are collected
                if len(st.session_state["sequence"]) == 30:
                    res = model.predict(np.expand_dims(st.session_state["sequence"], axis=0))[0]

                    # Append to sentence if confidence is high
                    if max(res) > 0.8:  # Confidence threshold
                        action = actions[np.argmax(res)]
                        confidence = max(res) * 100
                        if len(st.session_state["sentence"]) == 0 or action != st.session_state["sentence"][-1]:
                            st.session_state["sentence"].append(action)
                            st.session_state["accuracy"].append(f"{confidence:.2f}%")

                    # Visualize probabilities
                    prob_image = prob_viz(res, actions, crop_frame)
                    st.image(prob_image, caption="Gesture Recognition", channels="BGR")

            # Display results: Detected Gesture and Confidence
            st.write(f"Detected Gesture: {' '.join(st.session_state['sentence'])}")
            st.write(f"Confidence: {' '.join(st.session_state['accuracy'])}")

        else:
            st.error("Failed to capture a frame. Try again.")
    else:
        st.error("Failed to access the camera. Ensure it's connected.")
    cap.release()

# Stop button
stop_camera = st.button("Stop Camera")
if stop_camera:
    if "cap" in locals() and cap.isOpened():
        cap.release()
    st.success("Camera stopped.")
