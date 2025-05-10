import gtts
import os
import tempfile
from playsound import playsound

class WorkoutFeedback:
    def __init__(self):
        self.last_feedback = None

    def give_feedback(self, posture_label):
        message_map = {
            "correct": "Good form, keep it up!",
            "incorrect": "Please adjust your form.",
            "bend knees more": "Bend your knees more.",
            "keep spine straight": "Keep your spine straight.",
            "pose not fully visible": "Your full body is not visible, please adjust your position.",
            "start squatting": "Start your squat by bending your knees."
        }

        message = message_map.get(posture_label, None)
        if message and posture_label != self.last_feedback:
            self.speak(message)
            self.last_feedback = posture_label

    def speak(self, text):
        tmp_path = os.path.join(tempfile.gettempdir(), "feedback.mp3")
        tts = gtts.gTTS(text)
        tts.save(tmp_path)
        try:
            playsound(tmp_path)
        except Exception as e:
            print(f"Playback error: {e}")
        try:
            os.remove(tmp_path)
        except PermissionError:
            pass
