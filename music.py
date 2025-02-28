import os
import time
import logging
from typing import List, Tuple
from suno import Suno
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_music(music_client, prompt: str, output_path: str) -> AudioSegment:
    """
    Generate background music using Suno
    
    Args:
        music_client: Initialized Suno client
        prompt: Description of the music to generate
        output_path: Path to save the generated audio
        
    Returns:
        AudioSegment with the generated music
    """
    try:
        logger.info(f"Generating music for prompt: {prompt}")
        
        # Generate music
        generation = music_client.generate(
            prompt=prompt,
            duration_seconds=30,
            temperature=0.7
        )
        
        # Wait for generation to complete
        while not generation.is_done():
            logger.info("Waiting for music generation to complete...")
            time.sleep(5)
            
        # Download the generated audio
        output_file = generation.download_mp3(output_path)
        logger.info(f"Music generation complete: {output_file}")
        
        # Load as AudioSegment
        return AudioSegment.from_mp3(output_file)
    
    except Exception as e:
        logger.error(f"Error generating music: {e}")
        logger.warning("Returning silent audio due to music generation error")
        return AudioSegment.silent(duration=30000)  # 30 seconds of silence


def batch_music(music_client, base_path: str, music_descriptions: List[Tuple[int, str]]) -> List[Tuple[int, AudioSegment]]:
    """
    Generate multiple background music tracks based on descriptions
    
    Args:
        music_client: Initialized Suno client
        base_path: Base directory path
        music_descriptions: List of tuples containing (index, description)
        
    Returns:
        List of tuples containing (index, audio_segment)
    """
    # Temporary directory for music files
    temp_dir = os.path.join(base_path, "temp_music")
    os.makedirs(temp_dir, exist_ok=True)
    
    results = []
    
    for i, description in music_descriptions:
        output_file = os.path.join(temp_dir, f"music_{i}.mp3")
        
        # Generate the music
        audio = generate_music(music_client, description, output_file)
        results.append((i, audio))
        
    return results


if __name__ == "__main__":
    # Test music generation
    from dotenv import load_dotenv
    from suno import ModelVersions
    
    load_dotenv()
    
    suno_cookie = os.getenv('SUNO_COOKIE', '')
    music_client = Suno(cookie=suno_cookie, model_version=ModelVersions.CHIRP_V3_5)
    
    descriptions = [
        (1, "Suspenseful mystery music with violin and soft piano"),
        (2, "Energetic chase sequence with drums and brass")
    ]
    
    results = batch_music(music_client, "./", descriptions)
    print(f"Generated {len(results)} music tracks")