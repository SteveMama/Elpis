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

# Load environment variables from .env file
load_dotenv()

# Initialize API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client for Whisper API
client = OpenAI(api_key=OPENAI_API_KEY)


def listen_for_keyword():
    """
    Continuously listens for the wake word 'Indica' using speech recognition.

    Returns:
        bool: True if 'indica' is detected in the speech, False otherwise
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening for 'Indica'...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio).lower()
        return "indica" in text
    except:
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
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    st.write("Recording complete.")
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
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant helping a visually impaired person understand conversations and the world around them. You should translate foreign languages to English, describe situations, and provide context without assuming you're being directly addressed. Keep your answer very short and concise. the user is blind"
            },
            {
                "role": "user",
                "content": f"The user heard the following: '{text}'. If it's in a foreign language, translate it to English. Then, brief the user as to what was being spoken, as if explaining it to someone who can't see. Don't answer questions directly, just explain what was asked or said in less than 15 words."
            }
        ],
        "max_tokens": 1000
    }

    # Make API request to Groq
    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"


def speak_message(message):
    """
    Helper function to speak messages and handle audio playback.

    Args:
        message (str): Message to be spoken
    """
    tts = gTTS(text=message, lang='en')
    tts.save("message.mp3")
    st.audio("message.mp3", autoplay=True)
    # Add a small delay to ensure audio completes
    time.sleep(len(message.split()) * 0.3)
    os.remove("message.mp3")


def listen_for_keyword():
    """
    Continuously listens for the wake word 'Indica' using speech recognition.

    Returns:
        bool: True if 'indica' is detected in the speech, False otherwise
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.write("Listening for 'Indica'...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            text = recognizer.recognize_google(audio).lower()
            if "indica" in text:
                return True
            else:
                speak_message("I didn't hear Indica. Please say Indica when you need help.")
                return False
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




def text_to_speech(text):
    """
    Converts text to speech using Google's Text-to-Speech API.

    Args:
        text (str): Text to be converted to speech
    """
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    st.audio("response.mp3", autoplay=True)
    # Add delay based on text length
    time.sleep(len(text.split()) * 0.3)
    os.remove("response.mp3")


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