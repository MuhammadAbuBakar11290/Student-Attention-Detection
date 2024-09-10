import cv2
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

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

# Initialize counters and timers
start_time = datetime.now()
end_time = start_time + timedelta(minutes=2)

# Counters for each state
yawning_count = 0
focused_count = 0
unfocused_count = 0
total_frames = 0

# Define file paths
csv_filename = 'F:/Projects ML/Attentive  Detection/report.csv'
chart_filename = 'F:/Projects ML/Attentive Detection/report_chart.png'

# Initialize CSV file
header_written = False

try:
    # Real-time detection loop
    while datetime.now() < end_time:
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
        
        # Update counters based on predictions
        if predicted_yawn_class == 1:  # Assuming the class '1' represents yawning
            yawning_count += 1
        
        attention_label = attention_class_labels[predicted_attention_class]
        if attention_label == 'Straight':
            focused_count += 1
        else:
            unfocused_count += 1
        
        total_frames += 1

        # Display the prediction on the video feed
        if attention_label == 'Straight':
            cv2.putText(frame, 'Attentive', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Not Attentive', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Only show 'Yawning' if detected
        if predicted_yawn_class == 1:
            cv2.putText(frame, 'Yawning', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        cv2.namedWindow('Real-time Attentiveness and Yawn Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Attentiveness and Yawn Detection', 800, 600)
        cv2.imshow('Real-time Attentiveness and Yawn Detection', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Periodically save the CSV file
        if not header_written:
            with open(csv_filename, 'w') as file:
                file.write('State,Percentage\n')
            header_written = True

        if total_frames % 100 == 0:  # Save every 100 frames for example
            percentage_yawning = (yawning_count / total_frames) * 100 if total_frames else 0
            percentage_focused = (focused_count / total_frames) * 100 if total_frames else 0
            percentage_unfocused = (unfocused_count / total_frames) * 100 if total_frames else 0
            with open(csv_filename, 'a') as file:
                file.write(f'Yawning,{percentage_yawning}\n')
                file.write(f'Focused,{percentage_focused}\n')
                file.write(f'Unfocused,{percentage_unfocused}\n')

except Exception as e:
    print(f"An error occurred during the real-time detection loop: {e}")

finally:
    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Finalize CSV with final percentages
    percentage_yawning = (yawning_count / total_frames) * 100 if total_frames else 0
    percentage_focused = (focused_count / total_frames) * 100 if total_frames else 0
    percentage_unfocused = (unfocused_count / total_frames) * 100 if total_frames else 0

    try:
        # Append final results to CSV
        with open(csv_filename, 'a') as file:
            file.write(f'Yawning,{percentage_yawning}\n')
            file.write(f'Focused,{percentage_focused}\n')
            file.write(f'Unfocused,{percentage_unfocused}\n')
        print(f"CSV file saved as {csv_filename}")

        # Generate and save a bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(['Yawning', 'Focused', 'Unfocused'], [percentage_yawning, percentage_focused, percentage_unfocused], color=['red', 'green', 'blue'])
        plt.xlabel('State')
        plt.ylabel('Percentage')
        plt.title('Percentage of Yawning, Focused, and Unfocused States')
        plt.tight_layout()
        plt.savefig(chart_filename)
        plt.close()  # Close the plot to free up resources
        print(f"Chart saved as {chart_filename}")

    except Exception as e:
        print(f"An error occurred while saving files: {e}")
