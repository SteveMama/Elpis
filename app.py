import streamlit as st
import sounddevice as sd
import numpy as np
import httpx
from gtts import gTTS
import os
from openai import OpenAI
from dotenv import load_dotenv
import speech_recognition as sr

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def listen_for_keyword():
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
    st.write("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    st.write("Recording complete.")
    return audio.flatten(), sample_rate

def transcribe_audio(audio, sample_rate):
    temp_file = "temp_audio.wav"
    import scipy.io.wavfile as wav
    wav.write(temp_file, sample_rate, audio)

    with open(temp_file, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
        print("transcript", str(transcript))
    os.remove(temp_file)

    return transcript, "en"

def get_llm_response(text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for blind people. You can translate, provide information, and answer questions. Always prioritize clarity and conciseness in your responses."},
            {"role": "user", "content": f"Respond to this input: '{text}'. If it's in a foreign language, translate it to English. If it's a question or request for information, provide a helpful answer. Keep your response brief and to the point."}
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
    st.audio("response.mp3", autoplay=True)
    os.remove("response.mp3")

def main():
    st.title("Indica: Voice Assistant for the Visually Impaired")

    if st.button("Start Listening for 'Indica'"):
        while True:
            if listen_for_keyword():
                st.write("Keyword detected! How can I assist you?")
                audio, sample_rate = record_audio()

                transcribed_text, detected_language = transcribe_audio(audio, sample_rate)
                st.write(f"You said: {transcribed_text}")

                response = get_llm_response(transcribed_text)
                st.write(f"Indica's response: {response}")

                text_to_speech(response)
                st.write("Listening for 'Indica' again...")
            else:
                st.write("Keyword not detected. Listening again...")

if __name__ == "__main__":
    main()