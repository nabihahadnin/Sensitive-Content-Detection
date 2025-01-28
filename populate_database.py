import os
import cv2
import torch
from torchvision import transforms
from model import SimpleCNN  # Import the trained model
import mysql.connector

# MySQL database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Application@123',
    'database': 'video_library'
}

# Directory containing video files
VIDEO_FOLDER_PATH = 'C:/Users/vamsh/OneDrive/Documents/My Documents/Fall 2024 Courses/Advance Storage Processing and Retrieving of Big Data/Project/Active_Version/static/videos'

# Load the trained model
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load("sensitive_classifier.pth"))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to classify a frame using the ML model
def classify_frame(frame):
    # Preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame_pil = transforms.ToPILImage()(frame)  # Convert to PIL image
    frame_tensor = transform(frame_pil).unsqueeze(0)  # Apply transformations

    # Perform inference
    with torch.no_grad():
        outputs = model(frame_tensor)  # Get raw logits from the model
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Convert to probabilities
        _, predicted = torch.max(outputs, 1)  # Get the predicted class

    # Log probabilities and prediction
    print(f"Probabilities: {probabilities.numpy()}, Predicted: {predicted.item()}")

    return predicted.item()  # Return the predicted class (0 or 1)

# Function to process a single video
def process_video(video_path, video_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print(f"Skipping invalid video: {video_path}")
        cap.release()
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Processing video: {video_path}")
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        for second in range(int(duration)):
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            success, frame = cap.read()

            if not success or frame is None:
                print(f"Skipping invalid frame at second {second}")
                continue

            # Save the frame for testing
            frame_path = f"test_frames/frame_{video_id}_{second}.jpg"
            cv2.imwrite(frame_path, frame)

            # Classify the frame
            is_sensitive = classify_frame(frame)

            # Debugging probabilities
            print(f"Frame at {second}s classified as: {'Sensitive' if is_sensitive == 1 else 'Not Sensitive'}")

            if is_sensitive == 1:
                query = """
                INSERT INTO frames (video_id, timestamp, is_sensitive)
                VALUES (%s, %s, %s)
                """
                cursor.execute(query, (video_id, second, True))
                print(f"Inserting sensitive frame: Video ID {video_id}, Timestamp {second}")

        connection.commit()
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        cap.release()

# Function to process all videos in the database
def manually_process_videos():
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM videos")
        videos = cursor.fetchall()

        for video in videos:
            video_path = os.path.join(VIDEO_FOLDER_PATH, video['name'])
            process_video(video_path, video['id'])

        print("Finished processing all videos.")
    except mysql.connector.Error as err:
        print(f"Database Error: {err}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == "__main__":
    manually_process_videos()
