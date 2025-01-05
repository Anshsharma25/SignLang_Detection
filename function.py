import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize Mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Function to perform hand detection using MediaPipe
def mediapipe_detection(image, model):
    """
    Processes an image using MediaPipe Hands model.

    Args:
        image (numpy.ndarray): The input image.
        model (mediapipe.solutions.hands.Hands): MediaPipe hands model.

    Returns:
        tuple: Processed image and results from the hands model.
    """
    # Convert the image from BGR to RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with the model
    results = model.process(image)
    
    # Convert the image back to BGR color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results

# Function to extract keypoints from hand landmarks
def extract_keypoints(results):
    """
    Extracts keypoints from MediaPipe hand landmarks.

    Args:
        results (mediapipe.solutions.hands.Hands): Results from the MediaPipe model.

    Returns:
        numpy.ndarray: Flattened array of hand keypoints or zeros if no hands are detected.
    """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract (x, y, z) coordinates for each hand landmark
            keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            return keypoints
    return np.zeros(21 * 3)  # Return an array of zeros if no landmarks are detected

# Define the path for exporting data
DATA_PATH = os.path.join('MP_DATA')

# Define an array of actions (A, B, C)
actions = np.array(['A', 'B', 'C'])

# Define the number of sequences
no_sequences = 30

# Define the sequence length
sequence_length = 30
