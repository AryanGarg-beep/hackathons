import torch
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import pyttsx3  

# Load Whisper model
model = whisper.load_model("base")

# Record audio function
def record_audio(filename, duration=5, samplerate=16000):
    print("🎤 Recording... Speak now!")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, samplerate, audio)
    print("✅ Recording finished!")

# Speech-to-text function
def speech_to_text(audio_file):
    print("⏳ Transcribing...")
    result = model.transcribe(audio_file)
    print("📝 Recognized Text:", result["text"])
    return result["text"]

# Text-to-speech using pyttsx3
def text_to_speech(text):
    print("🎙️ Generating speech...")
    engine = pyttsx3.init()  # Initialize pyttsx3 engine
    engine.say(text)  # Speak the text
    engine.runAndWait()  # Run the speech engine
    print("✅ Done speaking!")

# Run the program
if __name__ == "__main__":
    audio_file = "input.wav"
    record_audio(audio_file, duration=5)
    text = speech_to_text(audio_file)
    text_to_speech(text)

