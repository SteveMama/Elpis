import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import httpx
from gtts import gTTS
import os
import json
from google.cloud import speech

# Set up Groq API key
GROQ_API_KEY = "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"


# Function to record audio
def record_audio(duration=5, sample_rate=44100):
    st.write("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    st.write("Recording complete.")
    return audio, sample_rate


# Function to transcribe audio using Google Cloud Speech-to-Text
def transcribe_audio(audio, sample_rate):
    client = speech.SpeechClient()

    content = audio.tobytes()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    transcription = ""
    for result in response.results:
        transcription += result.alternatives[0].transcript

    return transcription


# Function to get LLM response from Groq
def get_llm_response(text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for blind people."},
            {"role": "user", "content": text}
        ],
        "max_tokens": 1000
    }

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"


# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    st.audio("response.mp3")
    os.remove("response.mp3")


# Main Streamlit app
def main():
    st.title("Voice Assistant for Blind People (using Groq and Google Cloud)")

    if st.button("Start Voice Interaction") or st.session_state.get("voice_interaction", False):
        st.session_state.voice_interaction = True

        audio, sample_rate = record_audio()

        text = transcribe_audio(audio, sample_rate)
        st.write(f"You said: {text}")

        response = get_llm_response(text)
        st.write(f"Assistant's response: {response}")

        text_to_speech(response)

        st.session_state.voice_interaction = False


if __name__ == "__main__":
    main()
