# database/logger.py

import sqlite3
import datetime
import time

DB_PATH = "database/user_data.db"

def init_db():
    for _ in range(3):  # retry up to 3 times
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pose TEXT,
                    reps INTEGER,
                    feedback TEXT,
                    duration_seconds REAL,
                    date TEXT
                )
            ''')
            conn.commit()
            conn.close()
            return
        except sqlite3.OperationalError as e:
            if 'locked' in str(e).lower():
                time.sleep(1)
            else:
                raise e

def log_session(pose, reps, feedback_list, duration):
    feedback_str = "; ".join(feedback_list)
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _ in range(3):  # retry up to 3 times
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (pose, reps, feedback, duration_seconds, date)
                VALUES (?, ?, ?, ?, ?)
            ''', (pose, reps, feedback_str, duration, date))
            conn.commit()
            conn.close()
            return
        except sqlite3.OperationalError as e:
            if 'locked' in str(e).lower():
                time.sleep(1)
            else:
                raise e

def get_all_sessions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sessions ORDER BY date DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows
