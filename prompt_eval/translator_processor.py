import json
import requests
from typing import Dict, List, Any, Optional, Tuple


class TranslationProcessor:
    """
    A translation processor for GDPR compliance translations.
    Handles text preprocessing and translation via API calls.
    """
    
    def __init__(self, 
                 model: Optional[str] = None,
                 endpoint: str = "http://localhost:11434/v1/chat/completions",
                 timeout: int = 30,
                 temperature: Optional[float] = None,
                 prompt_path: Optional[str] = None):
        """
        Initialize the TranslationProcessor.
        
        Args:
            model (Optional[str]): Model name for translation (if None, must be specified in method calls)
            endpoint (str): API endpoint URL
            timeout (int): Request timeout in seconds
            temperature (Optional[float]): Model temperature for translation
            prompt_path (Optional[str]): Path to JSON file containing prompts
        """
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.temperature = temperature
        self.prompt_path = prompt_path
        
        # Cached prompts to avoid repeated file I/O
        self._cached_prompts: Optional[Tuple[str, str, str]] = None
        
        # Key mappings for data structure transformation
        self.key_mapping = {
            'article_number': 'numero_articolo'
        }
        
        # Language names for translation prompts
        self.language_names = {
            'it': 'Italian',
            'en': 'English', 
            'fr': 'French',
            'es': 'Spanish'
        }
    
    def load_prompt(self, file_path: str, 
                    system_key: str = "system_prompt", 
                    user_key: str = 'user_prompt',
                    assistant_key: str = 'assistant_prompt') -> Tuple[str, str, str]:
        """
        Load prompts from JSON file with caching.
        """
        # Return cached prompts if available
        if self._cached_prompts is not None:
            return self._cached_prompts
            
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Cache and return prompts
        self._cached_prompts = (data[system_key], data[user_key], data[assistant_key])
        return self._cached_prompts
    
    def substitute_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute dictionary keys with mapped equivalents.
        """
        new_data = {}
        for old_key, value in data.items():
            new_key = self.key_mapping.get(old_key, old_key)
            new_data[new_key] = value
        
        return new_data
    
    def translate(self, 
                 source_language: str,
                 target_language: str,
                 sentence: str,
                 model: Optional[str] = None) -> str:
        """
        Translate text using the configured API endpoint.
        """
        # Use provided model or instance model
        model_to_use = model or self.model
        if not model_to_use:
            return "Error: No model specified"
        
        if not self.prompt_path:
            return "Error: No prompt_path specified"
        
        # Load prompts (cached after first call)
        system_prompt, user_prompt, assistant_prompt = self.load_prompt(self.prompt_path)

        formatted_prompt = user_prompt.format(
            source_language=source_language,
            target_language=target_language,
            sentence=sentence
        )

        # Prepare payload
        payload = {
            "model": model_to_use,
            "messages": [
                {'role': "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt},
                {"role": "assistant", "content": assistant_prompt}
            ]
        }
        
        # Add temperature only if specified
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            return content.strip() if content else "Error: Empty response"
            
        except requests.exceptions.RequestException as e:
            return f"Connection error: {e}"
        except (KeyError, IndexError):
            return "Error: Invalid response format"
        except Exception as e:
            return f"Error: {e}"
    
    def process_evaluation_set(self,
                            eval_set: List[Dict[str, Any]],
                            source_lang: str,
                            target_langs: List[str],
                            model: Optional[str] = None,
                            apply_key_substitution: bool = True) -> List[Dict[str, Any]]:
        """
        Process evaluation set and generate translations.
        """
        # Use provided model or instance model
        model_to_use = model or self.model
        if not model_to_use:
            print("Error: No model specified")
            return []
        
        src2xxx = []
        
        # Process each item in evaluation set
        for el in eval_set:
            # Apply key substitution if requested
            processed_el = self.substitute_keys(el) if apply_key_substitution else el
            
            # Skip if source language not found
            if source_lang not in processed_el:
                continue
                
            sentence = processed_el[source_lang]
            source_language_name = self.language_names.get(source_lang, source_lang)
            
            # Generate translations for each target language
            for target_lang in target_langs:
                if target_lang not in processed_el:
                    continue
                    
                target_language_name = self.language_names.get(target_lang, target_lang)
                ref = processed_el[target_lang]
                translation = self.translate(source_language_name, target_language_name, 
                                           sentence, model=model_to_use)
                
                src2xxx.append({
                    'src': sentence, 
                    'mt': translation, 
                    'ref': ref
                })
        
        return src2xxx