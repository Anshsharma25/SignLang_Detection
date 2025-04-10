# 🤲 Sign Language Detection

Welcome to the **Sign Language Detection** project! This application uses computer vision and deep learning to detect and classify hand gestures for sign language recognition.

---

## 📖 Overview

This project leverages MediaPipe for hand detection and a custom-trained deep learning model to recognize various sign language gestures. It can predict hand gestures and display the corresponding actions or words in real-time. 
---

## ✨ Features

- 🖐️ Real-time hand tracking and gesture detection
- 🎯 High accuracy with a custom-trained model
- 📹 Uses your webcam for live video input
- 🖥️ User-friendly interface to display predictions

---

## 🛠️ Technologies Used

- **Programming Language**: Python 🐍
- **Libraries**: 
  - OpenCV 📷 (for video processing)
  - MediaPipe 🤖 (for hand tracking)
  - NumPy 🔢 (for numerical computations)
  - TensorFlow/Keras 🧠 (for deep learning model)
- **Model**: Custom-trained neural network

---

## 🚀 How to Run the Project

Follow these steps to set up and run the project locally:

### Prerequisites

1. Python 3.7 or higher 🐍
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Clone the Repository

```bash
git clone https://github.com/Anshsharma25/SignLang_Detection.git
cd SignLang_Detection
```

### Run the Application

1. Ensure your webcam is connected.
2. Execute the following steps in order:

   - **Step 1**: Run the helper functions for hand detection:
     ```bash
     python function.py
     ```
   - **Step 2**: (Optional) If training is needed, execute the training script:
     ```bash
     python training.py
     ```
   - **Step 3**: Run the gesture recognition system:
     ```bash
     python hand_landmark_prediction.py
     ```

3. The application will open a live video feed. Perform sign language gestures to see predictions in real-time.

---

## 📂 Project Structure

```
SignLang_Detection/
│
├── app.py                # Main application file
├── function.py           # Helper functions for hand tracking and keypoint extraction
├── training.py           # Script for training the gesture recognition model
├── hand_landmark_prediction.py # Script for testing hand gesture recognition
├── model.json            # JSON file of the trained model structure
├── model.h5              # Pre-trained model weights
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🎯 Process Workflow

1. **Hand Detection**: Uses MediaPipe to locate hand landmarks in the video feed.
2. **Keypoint Extraction**: Extracts coordinates of 21 hand landmarks.
3. **Gesture Recognition**: Processes the keypoints through a trained deep learning model to predict the gesture.
4. **Result Display**: Outputs the recognized gesture with its accuracy on the video feed.

---

## 📊 Model Training

The model was trained using a dataset of hand gesture images. Key steps included:
- Data collection for gestures (A, B, C, etc.)
- Preprocessing and augmentation
- Training a sequential neural network

----

## 🌟 Future Enhancements

- Add more gestures to the recognition system
- Support for multiple hand gestures simultaneously
- Deploy the application as a web-based service

---

## 👨‍💻 Author

Developed with ❤️ by [Ansh Sharma](https://github.com/Anshsharma25)

---


