# import streamlit as st
# import sounddevice as sd
# import numpy as np
# from faster_whisper import WhisperModel
# import tempfile
# import os
# import time

# # Initialize the Whisper model
# @st.cache_resource
# def load_model():
#     return WhisperModel("small", device="cpu", compute_type="int8")

# model = load_model()

# st.title("Live Speech Transcription with Faster Whisper")

# def record_audio(duration, sample_rate):
#     recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
#     sd.wait()
#     return recording

# # Streamlit UI
# sample_rate = 16000
# duration = 5  # Fixed duration for each recording

# st.write("Press 'Stop' to end the transcription.")

# # Start a loop for continuous recording and transcription
# if st.button("Start Live Transcription"):
#     st.write("Recording...")
#     while True:
#         audio_data = record_audio(duration, sample_rate)
        
#         # Save audio to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
#             temp_filename = temp_audio.name
#             import scipy.io.wavfile as wavfile
#             wavfile.write(temp_filename, sample_rate, audio_data)

#         st.write("Transcribing...")
#         segments, info = model.transcribe(temp_filename, beam_size=5)
        
#         st.write("Transcription:")
#         for segment in segments:
#             st.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

#         # Clean up the temporary file
#         os.unlink(temp_filename)
        
#         time.sleep(1)  # Optional: wait before the next recording

#         if st.button("Stop"):
#             st.write("Stopped recording.")
#             break

# st.write("Note: This app uses the CPU for transcription, which may be slower than GPU-based solutions.")



import streamlit as st
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import tempfile
import os
import time

# Initialize the Whisper model
@st.cache_resource
def load_model():
    return WhisperModel("small", device="cpu", compute_type="int8")

model = load_model()

st.title("Live Speech Transcription with Faster Whisper")

def record_audio(duration, sample_rate):
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return recording

# Streamlit UI
sample_rate = 16000
duration = 5  # Fixed duration for each recording cycle

st.write("Press 'Stop' to end the transcription.")

# Start a loop for continuous recording and transcription
if st.button("Start Live Transcription"):
    st.write("Recording...")
    while True:
        audio_data = record_audio(duration, sample_rate)
        
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_filename = temp_audio.name
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_filename, sample_rate, audio_data)

        st.write("Transcribing...")
        segments, info = model.transcribe(temp_filename, beam_size=5)
        
        st.write("Transcription:")
        for segment in segments:
            st.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        # Clean up the temporary file
        os.unlink(temp_filename)
        
        time.sleep(1)  # Optional: wait before the next recording

        # Assign a unique key to the Stop button
        if st.button("Stop", key=f"stop_button_{time.time()}"):
            st.write("Stopped recording.")
            break

st.write("Note: This app uses the CPU for transcription, which may be slower than GPU-based solutions.")
