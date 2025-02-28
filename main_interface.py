import streamlit as st
import json
import re
import os
import time
from dotenv import load_dotenv
from pathlib import Path
from gradio_client import Client
from suno import Suno, ModelVersions
# Import your functions here
from story import generate_story, get_llm_provider
from fuzzywuzzy import process
from music import batch_music
from xtts_helper import batch_tts
from stable_audio_testing import batch_effects
from pydub import AudioSegment
import concurrent.futures
from TTS.api import TTS

# Load environment variables from .env file
load_dotenv()

# Define paths using environment variables with fallbacks
base_path = os.getenv('BASE_PATH', os.path.join(os.path.dirname(__file__)))
speakers_path = os.getenv('SPEAKERS_PATH', os.path.join(base_path, "speakers"))
output_path = os.getenv('OUTPUT_PATH', os.path.join(base_path, "output"))

# Audio settings from environment
audio_fade_duration = int(os.getenv('AUDIO_FADE_DURATION_MS', 5000))
effect_volume_reduction = int(os.getenv('EFFECT_VOLUME_REDUCTION_DB', 5))
music_volume_reduction = int(os.getenv('MUSIC_VOLUME_REDUCTION_DB', 15))
output_format = os.getenv('OUTPUT_AUDIO_FORMAT', 'wav')

# Default LLM provider
default_provider = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')

# Ensure directories exist
os.makedirs(speakers_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# Function to get speaker files
def get_speaker_files():
    if not os.path.exists(speakers_path):
        return []
    speaker_files = [f for f in os.listdir(speakers_path) if f.endswith('.wav')]
    return [os.path.splitext(f)[0] for f in speaker_files]

# Function to get available LLM providers
def get_available_llm_providers():
    providers = []
    
    if os.getenv("OPENAI_API_KEY"):
        providers.append("OpenAI")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("Anthropic")
    if os.getenv("GEMINI_API_KEY"):
        providers.append("Gemini")
    
    return providers or ["OpenAI"]  # Default to OpenAI if none found

def parse_json(json_script):
    # Parse JSON script
    script = json_script['script']
    sound_effects_json = json_script['sound effects']
    background_music_json = json_script['background music']

    return script, sound_effects_json, background_music_json

def parse_script(script, sound_effects_json,background_music_json):
    chunks = re.split(r'(\[.*?\]|\{.*?\})', script)

    # Separate the chunks into their respective categories
    script_parts = []
    sound_effects = []
    background_music = []

    for i,chunk in enumerate(chunks):
        if chunk.startswith('[') and chunk.endswith(']'):
            #match with sound effects from the json
            match = process.extractOne(chunk[1:-1], sound_effects_json.keys())
            sound_effects.append((i,sound_effects_json[match[0]]['description']))
        elif chunk.startswith('{') and chunk.endswith('}'):
            #match with background music from the json
            match = process.extractOne(chunk[1:-1], background_music_json.keys())
            background_music.append((i,background_music_json[match[0]]['description']))
        else:
            script_parts.append((i,chunk))

    return script_parts, sound_effects, background_music

def generation_dispatcher(base_path, audio_client, music_client, script_parts, sound_effects, background_music, speaker_path):
    with concurrent.futures.ProcessPoolExecutor() as process_executor, \
         concurrent.futures.ThreadPoolExecutor() as thread_executor:
        
        # Submit the CPU-bound TTS task with speaker_path
        tts_future = process_executor.submit(batch_tts, base_path, script_parts, speaker_path)
        
        music_future = thread_executor.submit(batch_music, music_client, base_path, background_music)
        effects_future = thread_executor.submit(batch_effects, audio_client, base_path, sound_effects)
        
        music_result = music_future.result()
        effects_result = effects_future.result()
        tts_result = tts_future.result()

    return tts_result, music_result, effects_result

def assemble_audiobook(tts_segments, effect_segments, music_segments):
    # Combine and sort all segments
    all_segments = (
        [(i, segment, 'tts') for i, segment in tts_segments] +
        [(i, None, 'effect') for i, _ in effect_segments] +
        [(i, None, 'music') for i, _ in music_segments]
    )
    all_segments.sort(key=lambda x: x[0])
    final_audio = AudioSegment.silent(duration=0)
    effect_positions = {}
    music_positions = []
    current_position = 0

    for index, segment, seg_type in all_segments:
        if seg_type == 'tts':
            final_audio += segment
            current_position += len(segment)
        elif seg_type == 'effect':
            # Add a gap for the sound effect
            final_audio += AudioSegment.silent(duration=audio_fade_duration)
            effect_positions[index] = current_position
            current_position += audio_fade_duration
        elif seg_type == 'music':
            music_positions.append((index, current_position))

    # Overlay the sound effects
    for index, effect in effect_segments:
        if index in effect_positions:
            position = effect_positions[index]
            effect = effect.fade_in(audio_fade_duration).fade_out(audio_fade_duration)
            effect = effect - effect_volume_reduction  # Reduce volume
            final_audio = final_audio.overlay(effect, position=position)

    # Prepare and overlay the music segments
    music_positions.append((float('inf'), len(final_audio)))  # Add end position
    for i, (index, start_pos) in enumerate(music_positions[:-1]):
        _, end_pos = music_positions[i + 1]
        duration = end_pos - start_pos
        
        music = music_segments[i][1]
        music = music[:duration]  # Trim the music to the required duration
        
        # Apply fade in and fade out
        music = music.fade_in(audio_fade_duration).fade_out(audio_fade_duration)
        
        # Lower the volume of background music
        music = music - music_volume_reduction
        
        final_audio = final_audio.overlay(music, position=start_pos)

    return final_audio

def init_clients():
    # Init audio effects client
    stable_audio_url = os.getenv('STABLE_AUDIO_CLIENT_URL', '')
    audio_client = Client(stable_audio_url)
    
    # Init music client
    suno_cookie = os.getenv('SUNO_COOKIE', '')
    music_client = Suno(
        cookie=suno_cookie,
        model_version=ModelVersions.CHIRP_V3_5)
    
    return audio_client, music_client

def main():
    st.title("Audiobook Generation App")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    
    # LLM provider selection
    available_providers = get_available_llm_providers()
    selected_provider = st.sidebar.selectbox(
        "Select LLM Provider",
        available_providers,
        index=available_providers.index("OpenAI") if "OpenAI" in available_providers else 0
    )
    
    # Convert provider name to lowercase for API usage
    provider_name = selected_provider.lower()
    
    # Input section for story prompt
    st.header("Story Information")
    prompt = st.text_area("Enter your story prompt:", height=100)

    # Speaker selection
    st.header("Voice Selection")
    speakers = get_speaker_files()
    
    # Information about voice samples
    st.info("💡 You can use as little as a 5-second voice sample to clone any voice. Just upload a WAV file to the speakers directory.")
    
    if not speakers:
        st.warning("No speaker files found. Please add WAV files to the speakers directory.")
        selected_speaker = None
        
        # Add upload functionality for speakers
        uploaded_file = st.file_uploader("Upload a voice sample (WAV format)", type=["wav"])
        if uploaded_file:
            # Create a safe filename
            safe_filename = re.sub(r'[^\w]', '_', uploaded_file.name)
            speaker_path = os.path.join(speakers_path, safe_filename)
            
            # Save the uploaded file
            with open(speaker_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            st.success(f"Voice sample '{safe_filename}' uploaded successfully!")
            speakers = get_speaker_files()
            selected_speaker = os.path.splitext(safe_filename)[0]
    
    if speakers:
        selected_speaker = st.selectbox("Select a speaker:", speakers)
    
    try:
        audio_client, music_client = init_clients()
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        st.info("Please check your API credentials in the .env file")
        return

    if st.button("Generate Audiobook"):
        if prompt and selected_speaker:
            start_time = time.time()
            prompt += "\n Try to use only 2 to 3 background music and 2 to 4 sound effects. Stick to the json format"
            
            try:
                with st.spinner(f"Generating story using {selected_provider}..."):
                    json_string = generate_story(prompt, provider_name)
                    json_string = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_string)
                    json_script = json.loads(json_string)

                # Display the generated story
                st.subheader("Generated Story")
                st.json(json_script)

                # Parse the script
                script, sound_effects_json, background_music_json = parse_json(json_script)
                script_parts, sound_effects, background_music = parse_script(script, sound_effects_json, background_music_json)

                # Display parsed sections
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Script Parts")
                    st.write(script_parts)
                with col2:
                    st.subheader("Sound Effects")
                    st.write(sound_effects)
                with col3:
                    st.subheader("Background Music")
                    st.write(background_music)

                # Generate audiobook
                with st.spinner("Generating audiobook... This may take a while."):
                    progress_bar = st.progress(0)
                    speaker_path = os.path.join(speakers_path, f"{selected_speaker}.wav")
                    tts, music, effects = generation_dispatcher(base_path, audio_client, music_client, script_parts, sound_effects, background_music, speaker_path)
                    progress_bar.progress(100)

                # Assemble the audiobook
                with st.spinner("Assembling final audiobook..."):
                    final_audio = assemble_audiobook(tts, effects, music)
                    
                    # Create a sanitized filename from the prompt
                    safe_filename = re.sub(r'[^\w\s]', '', prompt.strip()[:30])
                    safe_filename = safe_filename.replace(' ', '_')
                    output_file = os.path.join(output_path, f"{safe_filename}_{int(time.time())}.{output_format}")
                    
                    final_audio.export(output_file, format=output_format)

                # Display the audio
                st.subheader("Generated Audiobook")
                st.audio(output_file)
                
                # Provide download link
                with open(output_file, "rb") as file:
                    btn = st.download_button(
                        label="Download Audiobook",
                        data=file,
                        file_name=os.path.basename(output_file),
                        mime=f"audio/{output_format}"
                    )
                
                end_time = time.time()
                time_taken = end_time - start_time
                st.info(f"Total time taken to generate the audiobook: {time_taken:.2f} seconds")
                
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
        else:
            st.warning("Please enter a prompt and select a speaker to generate the audiobook.")

if __name__ == "__main__":
    main()