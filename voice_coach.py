# utils/voice_coach.py
import pyttsx3
import time

class VoiceCoach:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.last_feedback_time = {}
        self.feedback_interval = 5  # seconds

    def speak(self, message, tag):
        current_time = time.time()
        if tag not in self.last_feedback_time or current_time - self.last_feedback_time[tag] > self.feedback_interval:
            try:
                self.engine.say(message)
                self.engine.runAndWait()
                self.last_feedback_time[tag] = current_time
            except RuntimeError:
                pass  # Handle run loop errors gracefully
