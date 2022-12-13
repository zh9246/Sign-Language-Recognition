from function import *
from sklearn.model_selection import train_test_split # to split the data into two parts, one for training and one for testing
from keras.utils import to_categorical # to convert our labels into categories
from keras.models import Sequential # to initialize our neural network
from keras.layers import LSTM, Dense # to create our layers
from keras.callbacks import TensorBoard # to generate TensorBoard logs
label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)
sequences, labels = [], [] # empty list
for action in actions: # iterate over each action, one at a time
    for sequence in range(no_sequences): # iterate over each sequence for the current action
        window = [] # empty window
        for frame_num in range(sequence_length): # iterate over frames in the current sequence
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))) # load the data
            window.append(res) 
        sequences.append(window) # append the window to the list of sequences
        labels.append(label_map[action]) # append the label

X = np.array(sequences) # convert the sequences list to numpy array
y = to_categorical(labels).astype(int) # convert the labels list to numpy array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs') # Path to save TensorBoard log files
tb_callback = TensorBoard(log_dir=log_dir) # TensorBoard callback
model = Sequential() # Initialize the constructor
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63))) # Add an input layer
model.add(LSTM(128, return_sequences=True, activation='relu')) # Add a LSTM layer with 128 internal units.
model.add(LSTM(64, return_sequences=False, activation='relu')) # Add a LSTM layer with 64 internal units.
model.add(Dense(64, activation='relu')) # Add a Dense layer with 64 internal units.
model.add(Dense(32, activation='relu')) # Add a Dense layer with 32 internal units.
model.add(Dense(actions.shape[0], activation='softmax')) # Add a Dense layer with as many units as actions: 3 in this case, and a softmax activation.
res = [.7, 0.2, 0.1] # 70% training, 20% validation, 10% test

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy']) # Compile the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback]) # Train the model
model.summary() # Print a summary of the model

model_json = model.to_json() # serialize model to JSON
with open("model.json", "w") as json_file:# serialize weights to HDF5
    json_file.write(model_json) # serialize model to JSON
model.save('model.h5') # serialize weights to HDF5