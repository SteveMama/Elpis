# Indica: Voice Assistant Documentation
A voice-activated assistant designed to help visually impaired users understand conversations and their surroundings through real-time translation and context interpretation.

## Technical Overview

### Core Technologies Used

1. **Speech Recognition and Processing**
   - `speech_recognition`: Wake word detection and voice command processing
   - `sounddevice`: Audio recording and playback
   - OpenAI's Whisper API: Multilingual speech-to-text conversion

2. **Language Processing**
   - Groq LLM API (llama3-8b-8192 model): Context understanding and translation
   - Natural language processing for multilingual support

3. **Text-to-Speech**
   - Google Text-to-Speech (gTTS): Converting responses to audio
   - Audio playback through Streamlit's audio component

4. **Frontend Interface**
   - Streamlit: Web-based user interface
   - Dynamic content updates using Streamlit containers

## System Architecture

### 1. Voice Activation System
```python
def listen_for_keyword():
    """
    Continuously listens for the wake word 'Indica'
    Returns: Boolean indicating wake word detection
    """
```
- Uses Google's Speech Recognition API
- Processes ambient audio through device microphone
- Triggers main functionality when "Indica" is detected

### 2. Audio Recording Module
```python
def record_audio(duration=5, sample_rate=16000):
    """
    Records user speech after wake word detection
    Parameters:
        duration: Recording length in seconds
        sample_rate: Audio quality parameter
    Returns: Audio data array and sample rate
    """
```
- Captures high-quality audio input
- Configurable duration and sample rate
- Returns normalized audio data

### 3. Speech-to-Text Processing
```python
def transcribe_audio(audio, sample_rate):
    """
    Converts speech to text using OpenAI's Whisper
    Parameters:
        audio: Recorded audio data
        sample_rate: Audio sample rate
    Returns: Transcribed text and language code
    """
```
- Utilizes OpenAI's Whisper model
- Supports multiple languages
- Handles temporary file management

### 4. Language Processing and Context Generation
```python
def get_llm_response(text):
    """
    Processes text through Groq's LLM
    Parameters:
        text: Input text for processing
    Returns: Contextual explanation or translation
    """
```
- Connects to Groq's API
- Processes input through LLM
- Generates contextual explanations
- Handles translations when needed

### 5. Text-to-Speech Output
```python
def text_to_speech(text):
    """
    Converts text responses to speech
    Parameters:
        text: Text to be converted to speech
    """
```
- Uses Google's TTS engine
- Generates clear, natural-sounding speech
- Handles audio file management

## Data Flow

1. **Input Phase**
   ```
   Wake Word Detection → Audio Recording → Speech-to-Text Conversion
   ```

2. **Processing Phase**
   ```
   Text Analysis → Language Detection → Translation (if needed) → Context Generation
   ```

3. **Output Phase**
   ```
   Response Formation → Text-to-Speech Conversion → Audio Playback
   ```

## Configuration and Setup

### Environment Variables
```
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Dependencies
```python
streamlit
sounddevice
numpy
httpx
gtts
openai
python-dotenv
speech_recognition
```

## Usage Instructions

1. **Installation**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**
   - Create .env file with API keys
   - Ensure microphone access is enabled

3. **Running the Application**
   ```bash
   streamlit run app.py
   ```

## Error Handling and Limitations

1. **Audio Input**
   - Handles microphone initialization failures
   - Manages audio recording exceptions

2. **API Connections**
   - Graceful handling of API timeouts
   - Error reporting for failed requests

3. **Known Limitations**
   - Requires stable internet connection
   - Fixed recording duration
   - Potential language detection delays

## Performance Considerations

1. **Response Time**
   - Average wake word detection: <1 second
   - Speech-to-text processing: 2-3 seconds
   - LLM processing: 1-2 seconds
   - Total interaction cycle: ~5-7 seconds

2. **Resource Usage**
   - Memory: ~200MB baseline
   - CPU: Moderate during audio processing
   - Network: ~1MB per interaction

## Security and Privacy

1. **Data Handling**
   - Temporary audio file storage
   - Secure API communications
   - No permanent data storage

2. **API Security**
   - Environment-based key management
   - Secure HTTPS connections
   - Rate limiting implementation

## Future Enhancements

1. **Planned Features**
   - Customizable wake word
   - Adjustable recording duration
   - Offline mode support
   - User preference storage

2. **Technical Improvements**
   - Enhanced error handling
   - Performance optimizations
   - Additional language support
   - Improved context awareness

This documentation provides a comprehensive overview of Indica's architecture, functionality, and implementation details, making it accessible for developers to understand and potentially contribute to the project.