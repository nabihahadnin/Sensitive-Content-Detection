import os
import cv2
import mysql.connector
import random  # For placeholder logic

# MySQL database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Application@123',
    'database': 'video_library'
}

# Directory containing video files
VIDEO_FOLDER_PATH = 'C:/Users/vamsh/OneDrive/Documents/My Documents/Fall 2024 Courses/Advance Storage Processing and Retrieving of Big Data/Project/Active_Version/static/videos'

# Placeholder function to simulate ML model prediction
def is_sensitive_placeholder(frame):
    return random.choice([0, 1])  # Randomly classify frames as sensitive or not

# Function to process video, extract frames, and classify sensitivity
def process_video(video_path, video_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Processing video: {video_path}")
    for second in range(int(duration)):
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)  # Move to the desired second
        success, frame = cap.read()

        if success:
            # Resize and normalize frame (if needed)
            processed_frame = cv2.resize(frame, (224, 224)) / 255.0

            # Use placeholder function to classify
            is_sensitive = is_sensitive_placeholder(processed_frame)

            if is_sensitive == 1:  # If frame is classified as sensitive
                store_sensitive_frame(video_id, second)

    cap.release()

# Function to store sensitive frame details in the database
def store_sensitive_frame(video_id, timestamp):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        query = """
        INSERT INTO frames (video_id, timestamp, is_sensitive)
        VALUES (%s, %s, %s)
        """
        cursor.execute(query, (video_id, timestamp, True))
        connection.commit()
        print(f"Stored sensitive frame: Video ID {video_id}, Timestamp {timestamp}")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Function to manually process videos
def manually_process_videos():
    # Connect to database and fetch videos
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
