import streamlit as st
import sounddevice as sd
import numpy as np
import httpx
from gtts import gTTS
import os
from openai import OpenAI
from dotenv import load_dotenv
import speech_recognition as sr
import time
import queue
from pydub import AudioSegment
from pydub.playback import play
import threading
import sys
import wave
import json
import io

# Load environment variables from .env file
load_dotenv()

# Initialize API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client for Whisper API
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize session state for maintaining context
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Initialize playback queue with priority
playback_queue = queue.PriorityQueue()

# Event to signal when the application is stopping
stop_event = threading.Event()

# Pre-generated voice notes and tones
voice_notes = {
    "welcome": "welcome_message.mp3",
    "recording_started": "recording_started.mp3",
    "recording_ended": "recording_ended.mp3",
    "listening_for_indica": "listening_for_indica.mp3",
    "heard_you": "heard_you.mp3",
    "still_listening": "still_listening.mp3",
    "didnt_catch": "didnt_catch.mp3",
    "error_microphone": "error_microphone.mp3",
}

tones = {
    "start_recording": "start_recording_tone.mp3",
    "end_recording": "end_recording_tone.mp3"
}

def play_audio(file_path):
    """
    Plays an audio file.

    Args:
        file_path (str): Path to the audio file to be played
    """
    try:
        audio = AudioSegment.from_file(file_path)
        play(audio)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"Error playing audio: {e}")

def play_next_message():
    if not playback_queue.empty():
        _, file_path = playback_queue.get()
        play_audio(file_path)
        st.session_state['audio_playing'] = True
        time.sleep(1)  # Ensure playback is finished before proceeding
        st.session_state['audio_playing'] = False
        play_next_message()

def speak_message(message_key, priority=1):
    """
    Helper function to speak messages and handle audio playback.

    Args:
        message_key (str): Key of the pre-generated message to be spoken
        priority (int): Priority level for the playback queue (lower number = higher priority)
    """
    playback_queue.put((priority, voice_notes[message_key]))
    if playback_queue.qsize() == 1:
        play_next_message()

def listen_for_keyword():
    """
    Continuously listens for the wake word 'Indica' using speech recognition.
    Allows interruption of ongoing playback.

    Returns:
        bool: True if 'indica' is detected in the speech, False otherwise
    """
    recognizer = sr.Recognizer()
    try:
        if st.session_state.get('audio_playing', False):
            # Stop playback if audio is currently playing
            st.session_state['audio_playing'] = False
            return True

        with sr.Microphone() as source:
            play_audio(voice_notes["listening_for_indica"])
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio).lower()

            if "indica" in text:
                return True

    except sr.WaitTimeoutError:
        speak_message("still_listening")
        return False
    except sr.UnknownValueError:
        speak_message("didnt_catch")
        return False
    except Exception as e:
        speak_message("error_microphone")
        print(f"Error: {str(e)}")
        return False

def record_audio(duration=5, sample_rate=16000):
    """
    Records audio input for a specified duration.

    Args:
        duration (int): Recording duration in seconds
        sample_rate (int): Audio sample rate in Hz

    Returns:
        tuple: (flattened audio array, sample rate)
    """
    play_audio(tones["start_recording"])
    speak_message("recording_started", priority=0)
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    play_audio(tones["end_recording"])
    speak_message("recording_ended", priority=0)
    return audio.flatten(), sample_rate

def transcribe_audio(audio, sample_rate):
    """
    Transcribes audio to text using OpenAI's Whisper API.

    Args:
        audio (numpy.array): Audio data
        sample_rate (int): Audio sample rate

    Returns:
        tuple: (transcribed text, language code)
    """
    temp_file = "temp_audio.wav"
    import scipy.io.wavfile as wav

    # Save audio to temporary WAV file
    wav.write(temp_file, sample_rate, audio)

    # Transcribe using Whisper API
    with open(temp_file, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        print("transcript", str(transcript))

    # Clean up temporary file
    os.remove(temp_file)

    # Add transcription to conversation history
    st.session_state.conversation_history.append({"role": "user", "content": transcript})

    return transcript, "en"

def get_llm_response(text):
    """
    Processes text input through Groq's LLM API for translation and context understanding.

    Args:
        text (str): Input text to be processed

    Returns:
        str: LLM's response with translation and context
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Define the conversation context and user prompt
    messages = st.session_state.conversation_history + [
        {
            "role": "system",
            "content": "You are an AI assistant helping a visually impaired person understand conversations and the world around them. You should translate foreign languages to English, describe situations, and provide context without assuming you're being directly addressed."
        },
        {
            "role": "user",
            "content": f"The user heard the following: '{text}'. If it's in a foreign language, translate it to English. Then, brief the user as to what was being spoken, as if explaining it to someone who can't see. Don't answer questions directly, just explain what was asked or said."
        }
    ]

    data = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "max_tokens": 1000
    }

    # Make API request to Groq
    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=data)

    if response.status_code == 200:
        llm_response = response.json()['choices'][0]['message']['content']
        # Add LLM response to conversation history
        st.session_state.conversation_history.append({"role": "assistant", "content": llm_response})
        return llm_response
    else:
        return f"Error: {response.status_code}, {response.text}"

def text_to_speech(text):
    """
    Converts text to speech using Google's Text-to-Speech API.

    Args:
        text (str): Text to be converted to speech
    """
    tts = gTTS(text=text, lang='en')
    response_file = "response.mp3"
    tts.save(response_file)
    play_audio(response_file)
    try:
        os.remove(response_file)
    except FileNotFoundError:
        print("Warning: response.mp3 was not found when attempting to delete.")

def main():
    """
    Main function that runs the Streamlit interface and manages the application flow.
    """
    st.title("Indica: Voice Assistant for the Visually Impaired")

    # Initialize session state for first run
    if 'first_run' not in st.session_state:
        st.session_state.first_run = True
        speak_message("welcome", priority=0)  # High priority welcome message

    # Create a placeholder for dynamic content updates
    placeholder = st.empty()

    # Main application loop
    while not stop_event.is_set():
        if listen_for_keyword():
            with placeholder.container():
                # Process voice input
                speak_message("heard_you", priority=1)
                audio, sample_rate = record_audio()

                # Convert speech to text
                transcribed_text, detected_language = transcribe_audio(audio, sample_rate)
                st.write(f"Heard: {transcribed_text}")

                # Get AI response
                response = get_llm_response(transcribed_text)
                st.write(f"Indica's explanation: {response}")

                # Convert response to speech
                text_to_speech(response)
                speak_message("listening_for_indica", priority=1)
        else:
            with placeholder.container():
                st.write("Listening for 'Indica'...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_event.set()
        print("Stopping the application...")
