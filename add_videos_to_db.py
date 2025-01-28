import os
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

# Function to add videos to the database
def add_videos_to_database():
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Get all video files from the folder
        video_files = [
            f for f in os.listdir(VIDEO_FOLDER_PATH)
            if f.endswith(('.mp4', '.avi', '.mov'))  # Supported formats
        ]

        for video_name in video_files:
            # Check if the video already exists in the database
            cursor.execute("SELECT * FROM videos WHERE name = %s", (video_name,))
            if cursor.fetchone():
                print(f"Video '{video_name}' already exists in the database. Skipping...")
                continue

            # Insert the video into the database
            query = "INSERT INTO videos (name, file_path) VALUES (%s, %s)"
            file_path = os.path.join(VIDEO_FOLDER_PATH, video_name)
            cursor.execute(query, (video_name, file_path))
            print(f"Added video: {video_name}")

        connection.commit()
        print("Finished adding all videos to the database.")

    except mysql.connector.Error as err:
        print(f"Database Error: {err}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == "__main__":
    add_videos_to_database()
