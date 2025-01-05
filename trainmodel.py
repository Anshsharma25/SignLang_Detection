# import necessary function and lib form the external file 'function.pu'
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical #explain?? >> use to convert labels in actual 
from keras.models import Sequential
from keras.layers import LSTM, Dense #lstm is use for converting the image into alphabitc 
from keras.callback import TensorBoard

# create a mapping of action labels to numercial values 
# (A>>>0 , B>>>>1 , C>>>>2)

label_map = {label: num for num, label in enumerate(actions)}

# Initialize empty lists to store sequences and labels
sequences, labels = [], []

#loop through each label(a,b,c)
for action in actions:
    # Loop through a fixed number of sequences for each action
    for sequence in range(no_sequences):
        window = []
        # Loop through a fixed sequence length
        for frame_num in range(sequence_length):
            # Load the stored keypoints for each frame in the sequence
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        # Append the sequence of keypoints to the sequences list
        sequences.append(window)
        # Append the numerical label for the action to the labels list
        labels.append(label_map[action])

#convert sequensce and label to numpy array 
X = np.array(sequence)
y = to_categorical(labels).astype(int)

#split the data into traing and testing set
X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.05)

#specfity the directory for tensorboard log (log dir)
log_dir = os.path.join('logs')

#create a tensorboard callback fro visualizing training progres
tb_callback = TensorBoard(log_dir=log_dir) # it helps to visuals the training process

#MODEL BUILDING
#initalize a sequential model
model = Sequential()

#add LSTM layer with specific conf
model.add(LSTM(64,return_sequences = True, activation ='relu', input_shape = (30,63)))  # 1 layer 
model.add(LSTM(128 ,return_sequences = True, activation ='relu'))# 2 layer 
model.add(LSTM(64 ,return_sequences = True, activation ='relu'))# 3 layer 

#add dense layer with specified conf
model.add(Dense(64 ,activation ='relu'))
model.add(Dense(32 ,activation ='relu'))

#add the final output layer with softmax acti for multiclass classfication
# it not used sigmod act is used for binary 0 & 1 
model.add(Dense(actions.shape[0] ,activation ='softmax'))

# Specify an example result array (not used)
# This array is provided as an example and is not used in the code. It represents
# a hypothetical set of prediction results for a given sequence, with three values.
res = [.7, 0.2, 0.1]

# Compile the model with specified optimizer, loss function, and evaluation metric
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model on the training data for a specified number of epochs, using the TensorBoard callback
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Display a summary of the model architecture
model.summary()

# Convert the model to JSON format and save it to a file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights to a separate file
model.save('model.h5')
