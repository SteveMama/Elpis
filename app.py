import streamlit as st
import sounddevice as sd
import numpy as np
import httpx
from gtts import gTTS
import os
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def record_audio(duration=5, sample_rate=16000):
    st.write("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    st.write("Recording complete.")
    return audio.flatten(), sample_rate


def transcribe_audio(audio, sample_rate):
    # Save the audio to a temporary file
    temp_file = "temp_audio.wav"
    import scipy.io.wavfile as wav
    wav.write(temp_file, sample_rate, audio)

    # Transcribe using OpenAI's Whisper API
    with open(temp_file, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file,response_format="text" )
        print("transcript", str(transcript))
    # Remove the temporary file
    os.remove(temp_file)

    return transcript, "en"  # Assuming English for simplicity


def get_llm_response(text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system",
             "content": "You are a helpful translator for the blind. "},
            {"role": "user", "content": f"for the given text: {text}, you must understand and convert it into simple english and let the user know what the other person is trying to say."}
        ],
        "max_tokens": 1000
    }

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"


def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    st.audio("response.mp3")
    os.remove("response.mp3")


def main():
    st.title("Voice Assistant with OpenAI Whisper and Groq LLM")

    if st.button("Start Voice Interaction") or st.session_state.get("voice_interaction", False):
        st.session_state.voice_interaction = True

        audio, sample_rate = record_audio()

        transcribed_text, detected_language = transcribe_audio(audio, sample_rate)
        st.write(f"Transcribed text: {transcribed_text}")

        context_response = get_llm_response(transcribed_text)
        st.write(f"Context and explanation: {context_response}")

        text_to_speech(context_response)

        st.session_state.voice_interaction = False


if __name__ == "__main__":
    main()