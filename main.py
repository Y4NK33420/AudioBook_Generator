import json
import re
# import torch
from fuzzywuzzy import process
from story import generate_story
from music import batch_music
from xtts_helper import batch_tts
from stable_audio_testing import batch_effects
from pydub import AudioSegment
import concurrent.futures
import os
from gradio_client import Client
from TTS.api import TTS
from suno import Suno, ModelVersions



#Defining global variables
base_path = os.getcwd()
base_path = r'D:\python projects\RAG\advanced rag\audiobook'



#function to handle the generation of script parts in a seperate thread


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

def generation_dispatcher(base_path, audio_client, music_client, script_parts, sound_effects, background_music):
    with concurrent.futures.ProcessPoolExecutor() as process_executor, \
         concurrent.futures.ThreadPoolExecutor() as thread_executor:
        
        # Submit the CPU-bound TTS task
        tts_future = process_executor.submit(batch_tts, base_path, script_parts)
        
        # Submit the IO worker task
        # io_worker_future = process_executor.submit(io_worker, base_path, audio_client, music_client, sound_effects, background_music)
        music_future = thread_executor.submit(batch_music, music_client, base_path, background_music)
        effects_future = thread_executor.submit(batch_effects, audio_client, base_path, sound_effects)
        
        music_result = music_future.result()
        effects_result = effects_future.result()
        # Wait for both tasks to complete and retrieve results
        tts_result = tts_future.result()
        # music_result, effects_result = io_worker_future.result()

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
            # Add a 3-second gap for the sound effect
            final_audio += AudioSegment.silent(duration=500)
            effect_positions[index] = current_position
            current_position += 500
        elif seg_type == 'music':
            music_positions.append((index, current_position))

    # Overlay the sound effects
    for index, effect in effect_segments:
        if index in effect_positions:
            position = effect_positions[index]
            fade_duration = min(500, len(effect) // 2)
            effect = effect.fade_in(fade_duration).fade_out(fade_duration)
            final_audio = final_audio.overlay(effect, position=position)

    # Prepare and overlay the music segments
    music_positions.append((float('inf'), len(final_audio)))  # Add end position
    for i, (index, start_pos) in enumerate(music_positions[:-1]):
        _, end_pos = music_positions[i + 1]
        duration = end_pos - start_pos
        
        music = music_segments[i][1]
        music = music[:duration]  # Trim the music to the required duration
        
        # Apply fade in and fade out (3 seconds each)
        fade_duration = 5000  # 3 seconds
        music = music.fade_in(fade_duration).fade_out(fade_duration)
        
        # Lower the volume of background music
        music = music - 15  # Reduce volume by 25 dB
        
        final_audio = final_audio.overlay(music, position=start_pos)

    return final_audio



if __name__ == '__main__':

    # Init audio effects client
    # Paste your client link here
    # audio_client = Client("https://f12ba7cedc27e58cd4.gradio.live/") just for example
    audio_client = Client() # link here
    print(f'Audio will be downloaded to {os.path.join(base_path,"audio_sample")}')
    
    # Init music client
    # Paste your suno cookies here
    music_client = Suno(
  cookie='',
  model_version=ModelVersions.CHIRP_V3_5)
    print(f'Music will be downloaded to {os.path.join(base_path,"music_sample")}')

    # prompt = input("Enter the prompt: ")
    prompt = "Generate a thriller story with twists and turns with a teenage protagonist with sound effects and music between the text as specified."

    json_string = generate_story(prompt)

    # Remove non-printable characters from JSON string
    json_string = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_string)
    json_script = json.loads(json_string)
    print(json_script)

    # Parse JSON script
    script, sound_effects_json, background_music_json = parse_json(json_script)
    print('-'*100)

    # Parse script
    script_parts, sound_effects, background_music = parse_script(script, sound_effects_json, background_music_json)

    print("script parts:", script_parts)
    print('-'*100, '\n\n')
    print("sound effects:", sound_effects)
    print('-'*100, '\n\n')
    print("background music:", background_music)

    # Generate the TTS, music, and effects
    tts, music, effects = generation_dispatcher(base_path,audio_client,music_client,script_parts, sound_effects, background_music)

    # Assemble the final audiobook
    final_audio = assemble_audiobook(tts, effects, music)

    # Save the final audio
    final_audio.export(r"D:\python projects\RAG\advanced rag\audiobook\title.wav", format="wav")