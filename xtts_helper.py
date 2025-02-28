import os
import logging
from pathlib import Path
import torch
import torchaudio
import numpy as np
from TTS.api import TTS
from pydub import AudioSegment
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_voice_sample(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load a voice sample file, handling files as short as 5 seconds
    
    Args:
        file_path: Path to the voice sample WAV file
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Voice sample file not found: {file_path}")
        
        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Convert to numpy array
        audio_array = waveform.numpy().squeeze()
        
        # Check duration and log
        duration = len(audio_array) / sample_rate
        logger.info(f"Loaded voice sample: {file_path} (duration: {duration:.2f}s, sample rate: {sample_rate}Hz)")
        
        return audio_array, sample_rate
    
    except Exception as e:
        logger.error(f"Error loading voice sample: {str(e)}")
        raise

def tts_fn(tts,base_path,text,file_name,speaker = r'D:\python projects\RAG\advanced rag\audiobook\speakers\female_speaker_1.wav'):
    file_path = os.path.join(base_path,file_name)
    try:
        if len(text) < 5:
            return AudioSegment.silent(duration=1000)
        else:
            return AudioSegment.from_wav(tts.tts_to_file(text=text, speaker_wav=speaker, language="en", file_path=file_path,split_sentences=True))
    except:
        print("Error in generating tts, returning silent audio")
        return AudioSegment.silent(duration=1000)

def batch_tts(base_path: str, script_parts: List[Tuple[int, str]], speaker_path: str) -> List[Tuple[int, AudioSegment]]:
    """
    Generate speech for all script parts using XTTS with voice cloning
    
    Args:
        base_path: Base directory path
        script_parts: List of tuples containing (index, text)
        speaker_path: Path to the speaker voice sample file
        
    Returns:
        List of tuples containing (index, audio_segment)
    """
    try:
        # Initialize TTS with XTTS model
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TTS model loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        # Temporary directory for audio files
        temp_dir = os.path.join(base_path, "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        
        results = []
        for i, part in script_parts:
            results.append((i, tts_fn(tts, base_path, part, os.path.join(temp_dir, f"output{i}.wav"), speaker_path)))
        
        return results
    
    except Exception as e:
        logger.error(f"Error in batch TTS: {str(e)}")
        raise

if __name__ == "__main__":
    import time
    text = """
The first step is to generate lyrics for the song.
We will be using chat G P T for this, enter your prompt and press enter.
Once your lyrics are generated, copy them as we will be using them for the song and video.
"""
    from TTS.api import TTS
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Init TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # tts.tts_to_file(text=text, speaker_wav=r"D:\python projects\RAG\advanced rag\her_target.wav", language="en", file_path=f"output.wav",split_sentences=True)
    t1 = time.time()
    #remove newline characters from the text
    segment = tts_fn(text.replace('\n',' '),'output')
    print("Time taken: ",time.time()-t1)
    #save the segment
    new_segment = segment + 3
    new_segment.export("pydub_1_enh.wav", format="wav")
    segment.export("pydub_1.wav", format="wav")
    #play audio using pydub
    # time.sleep(3)
    # play(segment)
