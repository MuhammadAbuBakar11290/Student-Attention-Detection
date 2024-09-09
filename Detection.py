# import cv2
# import numpy as np
# import os
# import sys
# from tensorflow.keras.models import load_model

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# sys.stdout.reconfigure(encoding='utf-8')
# os.environ['PYTHONIOENCODING'] = 'utf-8'

# # Load the pre-trained model
# model = load_model('attentive_yawn_model.h5')

# # Define class labels
# class_labels = ['Down', 'Left', 'Right', 'Straight', 'Up', 'Yawn']

# # Initialize webcam feed with a different backend (DirectShow for Windows)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Change to cv2.CAP_MSMF or cv2.CAP_GSTREAMER if needed

# # Image preprocessing function
# def preprocess_image(frame):
#     img_size = 224
#     img = cv2.resize(frame, (img_size, img_size))
#     img = img.astype('float32') / 255.0
#     img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input
#     return img

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame.")
#         break
    
#     # Preprocess the frame
#     preprocessed_frame = preprocess_image(frame)
    
#     try:
#         # Make predictions
#         predictions = model.predict(preprocessed_frame)
#         predicted_class = np.argmax(predictions[0])
        
#         # Display the prediction on the video feed
#         label = class_labels[predicted_class]
#         if label == 'Straight':
#             cv2.putText(frame, 'Attentive', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'Yawn':
#             cv2.putText(frame, 'Yawning', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         else:
#             cv2.putText(frame, 'Not Attentive', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
#     except UnicodeEncodeError as e:
#         print(f"UnicodeEncodeError occurred: {e}")
    
#     except Exception as e:
#         print(f"An error occurred: {e}")
    
#     # Display the frame
#     cv2.imshow('Real-time Attentiveness and Yawn Detection', frame)
    
#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release webcam and close windows
# cap.release()
# cv2.destroyAllWindows()
import cv2
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Load the pre-trained models for attention and yawning
attention_model = load_model('attention_model.h5')
yawn_model = load_model('yawn_model.h5')

# Define class labels for attention model
attention_class_labels = ['Down', 'Left', 'Right', 'Straight', 'Up']
# Only one relevant class label for the yawn model
yawn_class_labels = ['Yawning']

# Initialize webcam feed
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# Image preprocessing function
def preprocess_image(frame):
    img_size = 224
    img = cv2.resize(frame, (img_size, img_size))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input
    return img

# Real-time detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    preprocessed_frame = preprocess_image(frame)
    
    # Make predictions using both models
    attention_predictions = attention_model.predict(preprocessed_frame)
    yawn_predictions = yawn_model.predict(preprocessed_frame)
    
    # Get predicted class for attention
    predicted_attention_class = np.argmax(attention_predictions[0])
    # We only care about yawning, so if yawn detected:
    predicted_yawn_class = np.argmax(yawn_predictions[0])
    
    # Display the prediction on the video feed
    attention_label = attention_class_labels[predicted_attention_class]
    
    if attention_label == 'Straight':
        cv2.putText(frame, 'Attentive', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Not Attentive', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Only show 'Yawning' if detected
    if predicted_yawn_class == 1:  # Assuming the class '1' represents yawning
        cv2.putText(frame, 'Yawning', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('Real-time Attentiveness and Yawn Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

