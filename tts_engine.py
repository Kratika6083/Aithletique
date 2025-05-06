import pyttsx3
import threading

class VoiceCoach:
    def __init__(self, rate=160, volume=1.0, lang="english"):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self.lock = threading.Lock()

        # Optional: voice language config
        voices = self.engine.getProperty('voices')
        if lang == "hindi":
            for voice in voices:
                if "hi" in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        else:  # default to English
            for voice in voices:
                if "en" in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    break

    def speak(self, text):
        threading.Thread(target=self._speak_blocking, args=(text,)).start()

    def _speak_blocking(self, text):
        with self.lock:
            self.engine.say(text)
            self.engine.runAndWait()
