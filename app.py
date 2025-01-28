import os
from flask import Flask, render_template, send_from_directory, request, jsonify
import mysql.connector

app = Flask(__name__)

# Directory containing your video files
VIDEO_FOLDER_PATH = 'C:/Users/vamsh/OneDrive/Documents/My Documents/Fall 2024 Courses/Advance Storage Processing and Retrieving of Big Data/Project/Active_Version/static/videos'

# MySQL database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Application@123',
    'database': 'video_library'
}
connection = mysql.connector.connect(**db_config)

# Function to get all video files in the directory
def get_video_files():
    video_files = []
    for filename in os.listdir(VIDEO_FOLDER_PATH):
        if filename.endswith(('.mp4', '.avi', '.mov')):  # Supported video formats
            video_files.append({'url': f"/videos/{filename}", 'title': filename})
    return video_files

# Route to display the video gallery
@app.route('/')
def index():
    video_data = get_video_files()  # Get video data (URL and title)
    return render_template('index.html', video_data=video_data)

# Serve video files to the frontend
@app.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_FOLDER_PATH, filename)

# Fetch sensitive timestamps for a video
@app.route('/get_alerts', methods=['POST'])
def get_alerts():
    try:
        data = request.get_json()
        video_name = data['video_name']

        # Get video_id based on the name
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT id FROM videos WHERE name = %s", (video_name,))
        video = cursor.fetchone()

        if not video:
            return jsonify({'status': 'error', 'message': 'Video not found.'}), 404

        video_id = video['id']

        # Fetch sensitive timestamps
        cursor.execute("SELECT timestamp FROM frames WHERE video_id = %s AND is_sensitive = TRUE", (video_id,))
        timestamps = sorted([row['timestamp'] for row in cursor.fetchall()])

        # Consolidate continuous timestamps into start points
        consolidated_alerts = []
        for i in range(len(timestamps)):
            if i == 0 or timestamps[i] > timestamps[i - 1] + 1:
                consolidated_alerts.append(timestamps[i])

        return jsonify({'status': 'success', 'alerts': consolidated_alerts})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
