import torch
import whisper
import deepspeed
import sounddevice as sd
import scipy.io.wavfile as wav
import pyttsx3  # Import pyttsx3 for text-to-speech

# Load Whisper model with DeepSpeed optimization
model = whisper.load_model("base")
ds_model = deepspeed.init_inference(model, dtype=torch.float32)

# Function to record audio
def record_audio(filename, duration=5, samplerate=16000):
    print("🎤 Recording... Speak now!")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, samplerate, audio)
    print("✅ Recording finished!")

# Function to transcribe speech to text
def speech_to_text(audio_file):
    print("⏳ Transcribing...")
    result = model.transcribe(audio_file)
    print("📝 Recognized Text:", result["text"])
    return result["text"]

# Function to convert text to speech using pyttsx3
def text_to_speech(text):
    print("🎙️ Generating speech...")
    engine = pyttsx3.init(driverName='espeak')  # Initialize pyttsx3 engine
    engine.save_to_file(text, "output.wav")  # Save speech to file
    engine.runAndWait()  # Run the speech engine
    print("✅ Speech saved to 'output.wav'")
    
    # Play the audio
    import os
    os.system("play output.wav")



# Run the program
if __name__ == "__main__":
    audio_file = "input.wav"
    record_audio(audio_file, duration=5)
    text = speech_to_text(audio_file)
    text_to_speech(text)

