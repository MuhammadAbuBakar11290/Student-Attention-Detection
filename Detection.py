import cv2
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Environment setup
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Load models
attention_model = load_model('attention_model.h5')
yawn_model = load_model('yawn_model.h5')

# Define class labels
attention_class_labels = ['Down', 'Left', 'Right', 'Straight', 'Up']
yawn_class_labels = ['Yawning']

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Image preprocessing function
def preprocess_image(frame):
    img_size = 224
    img = cv2.resize(frame, (img_size, img_size))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
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
csv_filename = 'F:\Projects ML\Attentive Yawn Detection\report.csv'
bar_chart_filename = 'F:\Projects ML\Attentive Yawn Detection\bar_chart.png'
pie_chart_filename = 'F:\Projects ML\Attentive Yawn Detection\pie_chart.png'

# Function to update CSV and plots
def update_report():
    # Calculate percentages
    if total_frames > 0:
        percentage_yawning = (yawning_count / total_frames) * 100
        percentage_focused = (focused_count / total_frames) * 100
        percentage_unfocused = (unfocused_count / total_frames) * 100
    else:
        percentage_yawning = 0
        percentage_focused = 0
        percentage_unfocused = 0

    # Prepare data for CSV
    data = {
        'State': ['Attentive', 'Not Attentive', 'Yawning'],
        'Percentage': [f'{percentage_focused:.2f}%', f'{percentage_unfocused:.2f}%', f'{percentage_yawning:.2f}%']
    }

    try:
        # Save results to CSV
        df = pd.DataFrame(data)
        df.to_csv(csv_filename, index=False)
        print(f"CSV file saved as {csv_filename}")

        # Bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(df['State'], [percentage_focused, percentage_unfocused, percentage_yawning], color=['green', 'blue', 'red'])
        plt.xlabel('State')
        plt.ylabel('Percentage')
        plt.title('Percentage of Attentive, Not Attentive, and Yawning States')
        plt.tight_layout()
        plt.savefig(bar_chart_filename)
        plt.close()
        print(f"Bar chart saved as {bar_chart_filename}")

        # Pie chart
        plt.figure(figsize=(8, 8))
        plt.pie([percentage_focused, percentage_unfocused, percentage_yawning], 
                labels=df['State'], autopct='%1.1f%%', colors=['green', 'blue', 'red'], startangle=140)
        plt.title('Distribution of Attentive, Not Attentive, and Yawning States')
        plt.axis('equal')
        plt.savefig(pie_chart_filename)
        plt.close()
        print(f"Pie chart saved as {pie_chart_filename}")

    except Exception as e:
        print(f"An error occurred while saving files: {e}")

try:
    # Real-time detection loop
    while datetime.now() < end_time:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break
        
        # Preprocess frame and make predictions
        preprocessed_frame = preprocess_image(frame)
        attention_predictions = attention_model.predict(preprocessed_frame)
        yawn_predictions = yawn_model.predict(preprocessed_frame)
        
        # Get predicted class
        predicted_attention_class = np.argmax(attention_predictions[0])
        predicted_yawn_class = np.argmax(yawn_predictions[0])
        
        # Update counters
        if predicted_yawn_class == 1:
            yawning_count += 1
        
        attention_label = attention_class_labels[predicted_attention_class]
        if attention_label == 'Straight':
            focused_count += 1
        else:
            unfocused_count += 1
        
        total_frames += 1

        # Display predictions
        if attention_label == 'Straight':
            cv2.putText(frame, 'Attentive', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Not Attentive', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if predicted_yawn_class == 1:
            cv2.putText(frame, 'Yawning', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show frame
        cv2.namedWindow('Real-time Attentiveness and Yawn Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Attentiveness and Yawn Detection', 800, 600)
        cv2.imshow('Real-time Attentiveness and Yawn Detection', frame)
        
        # Update report after every frame
        update_report()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred during the real-time detection loop: {e}")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
