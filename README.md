# AI-Powered Audiobook Generator

A sophisticated audiobook generation system that creates realistic audiobooks from text prompts by orchestrating multiple AI models.

## Overview

This application generates complete audiobooks from simple text prompts using a pipeline of specialized AI models:

1. **LLM for Story Generation**: Creates the narrative structure, script, and identifies placement for audio effects and background music
2. **XTTS (Text-to-Speech)**: Converts the script into realistic speech using custom voice samples
3. **Stable Audio API**: Generates contextually appropriate sound effects
4. **Suno API**: Creates background music to enhance the storytelling experience

## Features

- **Interactive UI**: Simple Streamlit interface for prompt input and speaker selection
- **Custom Voice Support**: Use your own voice samples for narration
- **Voice Cloning**: Clone any voice with just a 5-second audio sample
- **Multiple LLM Providers**: Choose between different language models (OpenAI, Anthropic, Gemini, etc.)
- **Concurrent Generation**: Parallel processing of speech, music, and sound effects
- **Dynamic Assembly**: Intelligent mixing of all audio components with appropriate fades and volume adjustments
- **Complete Audiobook Experience**: Combines narration, background music, and sound effects into a cohesive audio narrative

## Requirements

- Python 3.8+
- API access to:
  - One or more LLM providers (OpenAI, Anthropic, Gemini, etc.)
  - Stable Audio API
  - Suno API
- Voice samples for XTTS (just 5 seconds is enough!)

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `example.env` to `.env` and fill in your API credentials:
   ```
   cp example.env .env
   ```
4. Add voice samples to the `speakers` directory (WAV format)

## Usage

1. Start the Streamlit application:
   ```
   streamlit run main_interface.py
   ```
2. Enter a prompt describing the audiobook you want to generate
3. Select a speaker voice from the dropdown
4. Click "Generate Audiobook" and wait for the process to complete
5. Listen to or download your generated audiobook

## Project Structure

```
audiobook/
├── main_interface.py     # Main Streamlit application
├── story.py              # LLM integration for story generation
├── xtts_helper.py        # Text-to-speech processing
├── music.py              # Background music generation via Suno
├── stable_audio_testing.py # Sound effects generation
├── .env                  # Environment variables (API keys, etc.)
├── speakers/             # Directory containing voice samples
└── output/               # Generated audiobooks are saved here
```

## Environment Variables

The following environment variables can be configured in your `.env` file:

- `BASE_PATH`: Root directory for the application
- `SPEAKERS_PATH`: Directory containing voice samples
- `STABLE_AUDIO_CLIENT_URL`: API endpoint for Stable Audio
- `SUNO_COOKIE`: Authentication cookie for Suno API
- `OPENAI_API_KEY`: API key for OpenAI (optional)
- `ANTHROPIC_API_KEY`: API key for Anthropic Claude (optional)
- `GEMINI_API_KEY`: API key for Google Gemini (optional)
- `DEFAULT_LLM_PROVIDER`: Default LLM to use (openai, anthropic, gemini, etc.)
- `AUDIO_FADE_DURATION_MS`: Duration for audio fades in milliseconds
- `EFFECT_VOLUME_REDUCTION_DB`: Volume reduction for sound effects in dB
- `MUSIC_VOLUME_REDUCTION_DB`: Volume reduction for background music in dB
- `OUTPUT_AUDIO_FORMAT`: Format for output files (e.g., "wav")

## Technical Details

The system operates in the following sequence:

1. **Story Generation**: LLM creates a structured JSON with script, sound effect markers, and background music markers
2. **Script Parsing**: The JSON is parsed to extract script parts, sound effect positions, and background music positions
3. **Parallel Generation**:
   - Text-to-speech converts script parts to audio
   - Sound effects are generated based on descriptions
   - Background music is created based on mood/scene descriptions
4. **Audio Assembly**: All components are mixed with appropriate timing, fades, and volume adjustments
5. **Output**: The final audiobook is presented to the user

## Limitations and Future Work

- Currently optimized for shorter stories (5-10 minutes)
- API rate limits may affect generation times
- Future enhancements:
  - Multi-character voice support
  - More granular control over audio mixing
  - Batch processing of multiple stories
  - Custom effect and music libraries

## Voice Sample Requirements

- Format: WAV file with 22050+ Hz sampling rate
- Duration: As little as 5 seconds is sufficient for voice cloning
- Quality: Clear audio without background noise works best
- Content: Sample should contain natural speech patterns

## Setup Instructions

### Running on Google Colab

1. Open the `Audiobook_helper.ipynb` notebook on Google Colab
2. Follow the installation steps in the notebook to set up Stable Audio Tools
3. Run the gradio server in the notebook
4. **Important:** Copy the public URL provided by gradio (looks like `https://xxxxx.gradio.live`)
5. Add this URL to your `.env` file as `STABLE_AUDIO_CLIENT_URL`

### Obtaining Suno Cookie

To get your Suno cookie for background music generation:

1. Go to [Suno](https://suno.ai) and log in to your account
2. Open your browser's developer tools (F12 or right-click → Inspect)
3. Go to the Network tab
4. Reload the page
5. Look for any request to suno.ai
6. Click on that request and find the "Cookie" header in the request headers
7. Copy the entire cookie string
8. Add this to your `.env` file as `SUNO_COOKIE="your-copied-cookie-value"`
