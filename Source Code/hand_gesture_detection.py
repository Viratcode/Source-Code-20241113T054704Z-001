import cv2 as cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model using TFSMLayer
model_path = "D:\\dowloads\\mp_hand_gesture-20241113T052258Z-001\\mp_hand_gesture"
model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

# Load class names
with open('D:\\dowloads\\Source Code-20241113T054704Z-001\\Source Code\\gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    if frame is None:
        print("Error: Failed to capture frame.")
        break

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    className = ''

    # Post-process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Convert landmarks to a numpy array (ensure it's in the correct shape)
            landmarks = np.array(landmarks, dtype=np.float32)

            # Debugging: Print landmarks to verify the shape and content
            print("Landmarks:", landmarks)

            # Predict gesture (using the TFSMLayer model)
            prediction = model(landmarks[None, ...])  # Add batch dimension

            # Debugging: Print the raw prediction to understand the model output
            print("Raw Prediction:", prediction)

            # If the prediction is a dictionary (like in TF Lite), extract it
            if isinstance(prediction, dict):
                prediction_data = prediction.get('predictions', None)
            else:
                prediction_data = prediction

            # Check if predictions were returned
            if prediction_data is None:
                print("No predictions returned. Check model input.")
                continue

            # Debugging: Print prediction data type and shape
            print("Prediction Data Type:", type(prediction_data))
            print("Prediction Data Shape:", prediction_data.shape if hasattr(prediction_data, 'shape') else "No shape attribute")

            # Convert tensor to numpy array if it's a tensor
            if isinstance(prediction_data, tf.Tensor):
                prediction_data = prediction_data.numpy()

            # Get the predicted class ID
            classID = np.argmax(prediction_data)
            className = classNames[classID]

    # Show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    # Check for key press event and exit if 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
