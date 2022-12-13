from function import * # import the function file
from keras.utils import to_categorical # to convert our labels into categories
from keras.models import model_from_json # to load the model
from keras.layers import LSTM, Dense # to create our layers
from keras.callbacks import TensorBoard # to generate TensorBoard logs
json_file = open("model.json", "r") # load json and create model
model_json = json_file.read() # load json and create model
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5") # load weights into new model

colors = [] # empty list
for i in range(0,20): # iterate over each action, one at a time
    colors.append((245,117,16)) # append the label
print(len(colors)) # print the length of the list
def prob_viz(res, actions, input_frame, colors,threshold): # function to visualize the probabilities
    output_frame = input_frame.copy() # make a copy of the input frame
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1) # draw the rectangle
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) # write the text
        
    return output_frame # return the output frame


# 1. New detection variables
sequence = []
sentence = []
accuracy=[]
predictions = []
threshold = 0.8 # probability threshold

cap = cv2.VideoCapture(0) # capture video
# cap = cv2.VideoCapture("https://192.168.43.41:8080/video")
# Set mediapipe model 
with mp_hands.Hands( # Hands is a class
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened(): # while the camera is open

        # Read feed
        ret, frame = cap.read() # read the camera feed

        # Make detections
        cropframe=frame[40:400,0:300] # crop the frame
        # print(frame.shape)
        frame=cv2.rectangle(frame,(0,40),(300,400),255,2) # draw the rectangle
        # frame=cv2.putText(frame,"Active Region",(75,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,255,2)
        image, results = mediapipe_detection(cropframe, hands) # make the mediapipe detection
        # print(results)
        
        # Draw landmarks
        # draw_styled_landmarks(image, results)
        # 2. Prediction logic
        keypoints = extract_keypoints(results) # extract the keypoints
        sequence.append(keypoints) # append the keypoints to the sequence
        sequence = sequence[-30:] # keep the last 30

        try: 
            if len(sequence) == 30: # if the sequence is 30
                res = model.predict(np.expand_dims(sequence, axis=0))[0] # predict the keypoints
                print(actions[np.argmax(res)]) # print the action
                predictions.append(np.argmax(res)) # append the prediction
                
                
            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res):  # if the last 10 predictions are the same
                    if res[np.argmax(res)] > threshold:        # and the probability is greater than the threshold
                        if len(sentence) > 0: # if the sentence is not empty
                            if actions[np.argmax(res)] != sentence[-1]: # and the action is not the same as the last action
                                sentence.append(actions[np.argmax(res)]) # append the action to the sentence
                                accuracy.append(str(res[np.argmax(res)]*100)) # append the accuracy to the sentence
                        else:
                            sentence.append(actions[np.argmax(res)]) # append the action to the sentence
                            accuracy.append(str(res[np.argmax(res)]*100)) # append the accuracy to the sentence

                if len(sentence) > 1: # if the sentence is greater than 1
                    sentence = sentence[-1:] # keep the last action
                    accuracy=accuracy[-1:] # keep the last accuracy

                # Viz probabilities
                # frame = prob_viz(res, actions, frame, colors,threshold)
        except Exception as e: 
            # print(e)
            pass
            
        cv2.rectangle(frame, (0,0), (300, 40), (245, 117, 16), -1) # draw the rectangle
        cv2.putText(frame,"Output: -"+' '.join(sentence)+''.join(accuracy), (3,30), # write the text
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # write the text
        
        # Show to screen
        cv2.imshow('OpenCV Feed', frame) # show the frame

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'): # if the key is q
            break
    cap.release()# release the camera
    cv2.destroyAllWindows() # destroy all the windows