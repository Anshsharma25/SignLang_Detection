#This is for the all helpers methods 
import cv2
import numpy as np
import os
import mediapipe as mp

#Iniatize Mediapipe drawing utiltites 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands 

#define the function to perform hand detection using media model 
def mediapipe_detection(image,model):
    
    #convert the imgae from BGR to RGb color space 
    image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
    
    # Make the images unWriteable to ensure its intergrity during processing
    image.flags.writable = False
    
    #Process the images to make  prediction
    results = model.process(image)
    
    #Make the imges writable again
    image.flags.writable = True
    
    #convert the image back to BGR color space
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    
    return image,results

def extract_keypoint(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
           rh = np.array(([res.x,res.y, res.z] for res in hand_landmarks.landmark)).flatten() if hand_landmarks else np.zeros(21 * 3)
           return np.concatenate([rh])
       
#define the path for exporint data
DATA_PATH = os.path.join('MP_DATA')

#Define an array of action (A<B<C)
actions = np.array(['A','B','C'])

#define the no of sequesnce 

no_sequences = 30 
sequnece_length = 30