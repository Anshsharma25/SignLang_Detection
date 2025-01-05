# entry point to our project 
from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense #lstm is use for converting the image into alphabitc 
from keras.callbacks import TensorBoard

#
json_file = open("model.json" ,"r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

#Define a list of color for visualization
color = []

#genrate 20 color and add them to the list 
for i in range(0,20):
    #each color is represented as tuple (r , g, b
    # where it is red green blue 
    #here , we can crating a orangis color with high red intensit(245)
    color.append((245,117,16))
print(len(color))

#define a fuction for visual probab on the output frame
#create the rectangle box for capture the hand and display the output to the user prob/confidence 
def prob_viz(res,actions , input_frame , threshold):
    output_frame = input_frame.copy()
    
    #loop through the prob and corresponfing actio 
    for num, prob in enumerate(res):
        #calucate postion for drwaing the rectangle and text
        #rectangle dimension (xi,y1) to (x2 , y2)
        
           x1 = 0
           y1 = 60 + num * 40 #giving some space b/w rectangel
           x2 = int(prob*100)# end the x cordinate
           y2 = 90 + (num * 40) # space b/w rectheight
           
           #draw a colored ractangle rep the prob
           cv2.rectangle(output_frame,{x1,y1},{x2,y2},color[num,-1])
           
           # put the action lable and prob on the frame 
           # action label psotion (test_x testy)
           text_x = 0 # x coordinate for the text label
           text_y = 85 + num*40 # y coordinate for the text label 
           
           cv2.putText(output_frame,actions[num],(text_x,text_y), cv2.FONT_HERSHEY_SIMPLEX,1 , (255,255,255), 2,cv2.LINE_AA)
           
           # Return the output frame with visualized probabilities
    return output_frame


# Initialize variables for gesture recognition
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8  # Set a threshold for gesture recognition confidence

# Open a video capture stream (0 represents the default camera)
cap = cv2.VideoCapture(0)
# You can also use an IP camera by providing a URL like:
# cap = cv2.VideoCapture("https://192.168.43.41:8080/video")

# Set up the MediaPipe Hands model with specific parameters
with mp_hands.Hands(
    model_complexity=0,  # Model complexity (0 for the fastest model)
    min_detection_confidence=0.5,  # Minimum confidence for hand detection
    min_tracking_confidence=0.5  # Minimum confidence for hand tracking
) as hands:
    while cap.isOpened():
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Crop the frame to a specific region of interest
        # Crop dimensions: Height (360 pixels) x Width (300 pixels)
        cropframe = frame[40:400, 0:300]

        # Draw a rectangle on the frame to highlight the region of interest
        # Rectangle coordinates: (x1, y1) to (x2, y2)
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)

        # Perform hand detection and tracking using MediaPipe
        image, results = mediapipe_detection(cropframe, hands)

        # Extract keypoints from the hand landmarks
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep only the last 30 frames for sequence input

        try:
            # Check if enough frames are collected for prediction
            if len(sequence) == 30:
                # Make a prediction using the loaded model
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])  # Print the recognized action

                # Store the prediction in the predictions list
                predictions.append(np.argmax(res))

                # Check if the predicted gesture has been consistent for the last 10 frames
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            # Append the recognized action to the sentence if it's different from the previous one
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)] * 100))
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)] * 100))

                # Keep only the most recent action and accuracy values
                if len(sentence) > 1:
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]

        except Exception as e:
            # Handle exceptions (e.g., if prediction fails)
            pass

        # Draw a colored rectangle for displaying the output sentence and accuracy
        # Rectangle coordinates: (x1, y1) to (x2, y2)
        # Draw a filled orange-colored rectangle at the top of the 'frame'.
        # The rectangle starts at (0, 0) as its top-left corner and ends at (300, 40).
        # It serves as a background for displaying output text.

        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)

        # Put the output text on the 'frame'.
        # The text "Output: -" followed by the 'sentence' and 'accuracy' content is displayed.
        # The text is positioned at (3, 30) on the frame.

        cv2.putText(frame, "Output: -" + ' '.join(sentence) + ''.join(accuracy), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        # Show the processed frame on the screen
        cv2.imshow('OpenCV Feed', frame)

        # Check for the 'q' key press to exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
