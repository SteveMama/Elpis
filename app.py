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
from pydub.generators import Sine
import threading

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

# Initialize playback queue
playback_queue = queue.Queue()

def generate_tone(file_name, frequency=440, duration_ms=500):
    """
    Generates a sine wave tone and saves it as an MP3 file.

    Args:
        file_name (str): Name of the file to save the tone.
        frequency (int): Frequency of the tone in Hz.
        duration_ms (int): Duration of the tone in milliseconds.
    """
    tone = Sine(frequency).to_audio_segment(duration=duration_ms)
    tone.export(file_name, format="mp3")

# Generate the tones if they do not exist
if not os.path.exists("start_recording_tone.mp3"):
    generate_tone("start_recording_tone.mp3", frequency=500, duration_ms=300)

if not os.path.exists("end_recording_tone.mp3"):
    generate_tone("end_recording_tone.mp3", frequency=600, duration_ms=300)

def play_next_message():
    if not playback_queue.empty():
        message = playback_queue.get()
        tts = gTTS(text=message, lang='en')
        tts.save("message.mp3")
        st.audio("message.mp3", autoplay=True)
        st.session_state['audio_playing'] = True
        time.sleep(len(message.split()) * 0.3)
        os.remove("message.mp3")
        st.session_state['audio_playing'] = False
        play_next_message()

def speak_message(message):
    """
    Helper function to speak messages and handle audio playback.

    Args:
        message (str): Message to be spoken
    """
    playback_queue.put(message)
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
            st.write("Listening for 'Indica'...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            text = recognizer.recognize_google(audio).lower()
            return "indica" in text
    except sr.WaitTimeoutError:
        speak_message("I'm still listening for Indica.")
        return False
    except sr.UnknownValueError:
        speak_message("I didn't catch that. Please say Indica clearly when you need help.")
        return False
    except Exception as e:
        speak_message("There was an error with the microphone. Please try again.")
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
    st.write("Recording...")
    play_tone("start_recording_tone.mp3")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    play_tone("end_recording_tone.mp3")
    st.write("Recording complete.")
    return audio.flatten(), sample_rate

def play_tone(tone_file):
    """
    Plays an auditory tone to signal different stages of the interaction.

    Args:
        tone_file (str): File path of the tone to be played
    """
    try:
        tone = AudioSegment.from_file(tone_file)
        play(tone)
    except FileNotFoundError:
        print(f"Error: The file {tone_file} was not found.")
    except Exception as e:
        print(f"Error playing tone: {e}")

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
    playback_queue.put(text)
    if playback_queue.qsize() == 1:
        play_next_message()

def main():
    """
    Main function that runs the Streamlit interface and manages the application flow.
    """
    st.title("Indica: Voice Assistant for the Visually Impaired")

    # Initialize session state for first run
    if 'first_run' not in st.session_state:
        st.session_state.first_run = True
        welcome_message = "Hi there, this is Indica, I can help you. For anything you need, say Indica and ask."
        speak_message(welcome_message)

    # Create a placeholder for dynamic content updates
    placeholder = st.empty()

    # Main application loop
    while True:
        if listen_for_keyword():
            with placeholder.container():
                # Process voice input
                speak_message("I heard you! What can I help you with?")
                audio, sample_rate = record_audio()

                # Convert speech to text
                transcribed_text, detected_language = transcribe_audio(audio, sample_rate)
                st.write(f"Heard: {transcribed_text}")

                # Get AI response
                response = get_llm_response(transcribed_text)
                st.write(f"Indica's explanation: {response}")

                # Convert response to speech
                text_to_speech(response)
                speak_message("I'm listening for Indica again.")
        else:
            with placeholder.container():
                st.write("Listening for 'Indica'...")

if __name__ == "__main__":
    main()
