# database.py

import mysql.connector
from config import DB_CONFIG


def get_connection():
    """Establish a connection to the database."""
    return mysql.connector.connect(**DB_CONFIG)


def fetch_sensitive_frames(video_id):
    """Fetch sensitive timestamps for a given video."""
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(
            "SELECT FLOOR(timestamp) AS timestamp FROM frames WHERE video_id = %s AND is_sensitive = TRUE",
            (video_id,)
        )
        return [row['timestamp'] for row in cursor.fetchall()]
    finally:
        cursor.close()
        connection.close()


def fetch_all_videos():
    """Fetch all video IDs from the database."""
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute("SELECT id FROM videos")
        return cursor.fetchall()
    finally:
        cursor.close()
        connection.close()
