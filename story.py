import os
import json
import re
import logging
from typing import Dict, Any, Optional
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider:
    """Base class for LLM providers"""
    def generate_completion(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement this method")


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation"""
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
    def generate_completion(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a creative storyteller who writes engaging stories with sound effects and background music cues. Output your response in valid JSON format with 'script', 'sound effects', and 'background music' keys."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation"""
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
    def generate_completion(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        
        try:
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                temperature=0.7,
                system="You are a creative storyteller who writes engaging stories with sound effects and background music cues. Output your response in valid JSON format with 'script', 'sound effects', and 'background music' keys.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation"""
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
    def generate_completion(self, prompt: str) -> str:
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        
        try:
            model = genai.GenerativeModel('gemini-pro')
            system_prompt = "You are a creative storyteller who writes engaging stories with sound effects and background music cues. Output your response in valid JSON format with 'script', 'sound effects', and 'background music' keys."
            
            response = model.generate_content(
                [system_prompt, prompt],
                generation_config={"temperature": 0.7, "max_output_tokens": 2000}
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


def get_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """
    Get the appropriate LLM provider instance
    
    Args:
        provider_name: The name of the provider to use. If None, uses the default from .env
        
    Returns:
        An instance of LLMProvider
    """
    if not provider_name:
        provider_name = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()
    
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider
    }
    
    if provider_name not in providers:
        logger.warning(f"Unknown provider '{provider_name}', falling back to OpenAI")
        provider_name = "openai"
    
    try:
        return providers[provider_name]()
    except Exception as e:
        logger.error(f"Error initializing {provider_name} provider: {e}")
        raise


def sanitize_json(text: str) -> str:
    """
    Attempt to extract and sanitize JSON from text
    """
    # Try to find JSON block if not already JSON
    if not text.strip().startswith("{"):
        json_match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # Try to find the first { and last } for a potential JSON object
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start:end+1]
    
    # Replace escaped characters
    text = text.replace('\\"', '"')
    return text


def generate_story(prompt: str, provider_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a story with sound effects and background music using the specified LLM provider
    
    Args:
        prompt: The story prompt
        provider_name: The LLM provider to use (openai, anthropic, gemini, etc.)
        
    Returns:
        JSON object with script, sound effects, and background music
    """
    start_time = time.time()
    
    # Get the appropriate LLM provider
    provider = get_llm_provider(provider_name)
    
    # Enhance the prompt for better output
    enhanced_prompt = f"""
    Create a short story based on this prompt: {prompt}

    Format your response as a valid JSON object with the following structure:
    {{
      "script": "The story text with [sound effect name] and {{background music description}} markers at appropriate moments",
      "sound effects": {{
        "sound effect name": {{
          "description": "detailed description for generating the sound"
        }},
        ...more sound effects...
      }},
      "background music": {{
        "music description": {{
          "description": "detailed description for generating the music"
        }},
        ...more background music...
      }}
    }}

    Use 2-4 sound effects and 1-3 background music changes total. Place sound effect markers like [footsteps] and background music changes like {{suspenseful}} directly within the script text.
    """
    
    try:
        # Generate the completion
        response = provider.generate_completion(enhanced_prompt)
        
        # Parse the response as JSON
        sanitized = sanitize_json(response)
        story_data = json.loads(sanitized)
        
        # Validate the structure
        required_keys = ["script", "sound effects", "background music"]
        for key in required_keys:
            if key not in story_data:
                raise ValueError(f"Missing required key in response: {key}")
        
        end_time = time.time()
        logger.info(f"Story generated in {end_time - start_time:.2f} seconds using {provider.__class__.__name__}")
        
        return story_data
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}\nResponse: {response}")
        raise ValueError("The LLM response could not be parsed as JSON")
    
    except Exception as e:
        logger.error(f"Error in generate_story: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the story generation
    test_prompt = "A short mystery story set in an old lighthouse"
    result = generate_story(test_prompt)
    print(json.dumps(result, indent=2))