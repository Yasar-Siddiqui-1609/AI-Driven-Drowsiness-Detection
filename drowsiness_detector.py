import cv2
import numpy as np
from keras.models import load_model # Using keras directly based on your earlier fix
from pygame import mixer

# 1. Initialize Audio Alarm
mixer.init()
mixer.music.load('alarm.wav')

# 2. Load your custom-trained ML model
model = load_model('drowsiness_model.h5')
IMG_SIZE = 80

# 3. Load OpenCV's built-in Haar Cascades (NO MEDIAPIPE NEEDED!)
# These XML files come pre-installed natively inside the opencv-python library
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Game logic variables
SCORE = 0  
THRESHOLD = 6  # Frames eyes must be closed to trigger the alarm
ALARM_ON = False

# Connect to Webcam
cap = cv2.VideoCapture(0)

print("Starting Real-Time Detection with Haar Cascades... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Haar Cascades require grayscale images to detect features
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    state = "Open" # Default state
    eyes_detected = False
    
    for (x, y, w, h) in faces:
        # Draw a blue rectangle around the face (Visual UI)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Isolate the face region to look for eyes (saves processing power)
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        
        for (ex, ey, ew, eh) in eyes:
            eyes_detected = True
            
            # Draw a green rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Crop the eye exactly like we trained the CNN
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_img_resized = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
            eye_img_normalized = eye_img_resized / 255.0
            eye_img_reshaped = np.reshape(eye_img_normalized, (-1, IMG_SIZE, IMG_SIZE, 1))
            
            # Ask the ML model to classify the eye
            prediction = model.predict(eye_img_reshaped, verbose=0)[0][0]
            
            # If the prediction is closer to 0, the eye is closed
            if prediction < 0.5:
                state = "Closed"
            else:
                state = "Open"
                
            # We only need to check one eye per frame to adjust the score
            break 
            
    # Update the fatigue score based on the AI's prediction
    if state == "Closed" and eyes_detected:
        SCORE += 1
    elif state == "Open" and eyes_detected:
        SCORE -= 1
        if SCORE < 0:
            SCORE = 0

    # Screen UI Updates
    cv2.putText(frame, f"Eyes: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Fatigue Score: {SCORE}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Audio/Visual Alert Logic
    if SCORE > THRESHOLD:
        cv2.putText(frame, "DROWSINESS DETECTED!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        if not ALARM_ON:
            mixer.music.play(-1) # Loop alarm
            ALARM_ON = True
    else:
        if ALARM_ON:
            mixer.music.stop()
            ALARM_ON = False

    cv2.imshow('AI Nexus - Drowsiness Detection (Haar Edition)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()