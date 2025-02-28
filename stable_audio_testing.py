import os
import logging
from typing import List, Tuple
from gradio_client import Client
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sound_effect(client, prompt: str, output_file: str) -> AudioSegment:
    """
    Generate a sound effect using Stable Audio
    
    Args:
        client: Initialized Gradio client for Stable Audio
        prompt: Description of the sound effect
        output_file: Path to save the generated audio
        
    Returns:
        AudioSegment with the generated sound effect
    """
    try:
        logger.info(f"Generating sound effect: {prompt}")
        
        # Generate the sound effect
        result = client.predict(
            prompt,  # Prompt
            50,      # Number of steps
            0.01,    # Seed
            True,    # Use custom seed
            api_name="/run"
        )
        
        # The result is a list with the generated audio file path
        generated_file = result[0]
        
        # Save to the output location
        if os.path.exists(generated_file):
            # Load the audio and convert if needed
            audio = AudioSegment.from_file(generated_file)
            audio.export(output_file, format="wav")
            logger.info(f"Sound effect saved to {output_file}")
            return audio
        else:
            logger.error(f"Generated file not found: {generated_file}")
            return AudioSegment.silent(duration=5000)
    
    except Exception as e:
        logger.error(f"Error generating sound effect: {e}")
        logger.warning("Returning silent audio due to sound effect generation error")
        return AudioSegment.silent(duration=5000)  # 5 seconds of silence


def batch_effects(client, base_path: str, effect_descriptions: List[Tuple[int, str]]) -> List[Tuple[int, AudioSegment]]:
    """
    Generate multiple sound effects based on descriptions
    
    Args:
        client: Initialized Gradio client for Stable Audio
        base_path: Base directory path
        effect_descriptions: List of tuples containing (index, description)
        
    Returns:
        List of tuples containing (index, audio_segment)
    """
    # Temporary directory for sound effect files
    temp_dir = os.path.join(base_path, "temp_effects")
    os.makedirs(temp_dir, exist_ok=True)
    
    results = []
    
    for i, description in effect_descriptions:
        output_file = os.path.join(temp_dir, f"effect_{i}.wav")
        
        # Generate the sound effect
        audio = generate_sound_effect(client, description, output_file)
        results.append((i, audio))
        
    return results


if __name__ == "__main__":
    # Test sound effect generation
    from dotenv import load_dotenv
    
    load_dotenv()
    
    stable_audio_url = os.getenv('STABLE_AUDIO_CLIENT_URL', '')
    client = Client(stable_audio_url)
    
    descriptions = [
        (1, "A loud, buzzing alarm clock sound"),
        (2, "Glass breaking with echoing sound")
    ]
    
    results = batch_effects(client, "./", descriptions)
    print(f"Generated {len(results)} sound effects")