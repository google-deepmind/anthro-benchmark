# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LLM client initialization and management for dialogue generation.
"""

from typing import Dict, Any, Optional
import os

class LLMClient:
    """Base class for LLM clients."""
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> 'LLMClient':
        """
        Factory method to create the appropriate LLM client.
        
        Args:
            config: LLM configuration dictionary
            
        Returns:
            Initialized LLM client
        """
        model_type = config.get('model', '').lower()
        
        if 'gpt' in model_type:
            return OpenAIClient(config)
        elif 'claude' in model_type:
            return AnthropicClient(config)
        elif 'mistral' in model_type:
            return MistralClient(config)
        elif 'gemini' in model_type:
            return GeminiClient(config)
        else:
            return DefaultClient(config)
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM client.
        
        Args:
            config: LLM configuration dictionary
        """
        self.config = config
        self.model = config.get('model', 'default')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 1000)
    
    def generate(self, messages: list, **kwargs) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            Generated text response
        """
        raise NotImplementedError("Subclasses must implement this method")


class OpenAIClient(LLMClient):
    """Client for OpenAI models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI client."""
        super().__init__(config)
        self.api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        try:
            import openai
            self.client = openai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required for OpenAI models")
    
    def generate(self, messages: list, **kwargs) -> str:
        """Generate response using OpenAI API."""
        try:
            params = {
                'model': self.model,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                **kwargs
            }
            
            response = self.client.chat.completions.create(
                messages=messages,
                **params
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise


class AnthropicClient(LLMClient):
    """Client for Anthropic models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Anthropic client."""
        super().__init__(config)
        self.api_key = config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package is required for Claude models")
    
    def generate(self, messages: list) -> str:
        """Generate response using Anthropic API."""
        try:
            model_name = self.model
            request_temperature = self.temperature
            request_max_tokens = self.max_tokens
            
            system_prompt_content = None 
            anthropic_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_prompt_content = msg['content'] 
                else:
                    role = msg['role']
                    if role == 'model': 
                        role = 'assistant'
                    anthropic_messages.append({
                        'role': role,
                        'content': msg['content']
                    })
            
            if system_prompt_content is not None:
                response = self.client.messages.create(
                    model=model_name, 
                    messages=anthropic_messages,
                    system=system_prompt_content, 
                    temperature=request_temperature, 
                    max_tokens=request_max_tokens   
                )
            else:
                response = self.client.messages.create(
                    model=model_name, 
                    messages=anthropic_messages,
                    temperature=request_temperature, 
                    max_tokens=request_max_tokens   
                )
            
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            raise


class MistralClient(LLMClient):
    """Client for Mistral AI models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Mistral client."""
        super().__init__(config)
        self.api_key = config.get('api_key') or os.environ.get('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("Mistral API key is required")
        
        try:
            from mistralai import Mistral 
            self.client = Mistral(api_key=self.api_key)
        except ImportError:
            raise ImportError("mistralai package is required for Mistral models. Please install it (e.g., pip install mistralai).")
    
    def generate(self, messages: list, **kwargs) -> str:
        """Generate response using Mistral API (new client version)."""
        try:
            processed_messages = []
            for msg in messages:
                role = msg['role']
                if role == 'model': 
                    role = 'assistant'
                processed_messages.append({'role': role, 'content': msg['content']})

            params = {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
            }

            for k, v in kwargs.items():
                if k not in ['model', 'messages']:
                    params[k] = v
            
            response = self.client.chat.complete(
                model=self.model,
                messages=processed_messages,
                **params
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Mistral API: {e}")
            raise
        
class GeminiClient(LLMClient):
    """Client for Google Gemini models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Gemini client."""
        super().__init__(config)
        self.api_key = config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini models")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except ImportError:
            raise ImportError("google-generativeai package is required for Gemini models")
    
    def generate(self, messages: list, **kwargs) -> str:
        """Generate response using Gemini API."""
        try:
            gen_config_params = {
                'temperature': self.temperature,
                'max_output_tokens': self.max_tokens,
            }
            for k, v in kwargs.items():
                 gen_config_params[k] = v

            generation_config = self.genai.types.GenerationConfig(**gen_config_params)
            
            gemini_messages_for_api = []
            system_prompt_content = None
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_prompt_content = msg['content']
                elif msg['role'] == 'user':
                    gemini_messages_for_api.append({"role": "user", "parts": [msg['content']]})
                elif msg['role'] == 'assistant' or msg['role'] == 'model':
                    gemini_messages_for_api.append({"role": "model", "parts": [msg['content']]})
            
            model_kwargs = {}
            if system_prompt_content:
                model_kwargs['system_instruction'] = system_prompt_content
            
            model = self.genai.GenerativeModel(
                model_name=self.model, 
                **model_kwargs
            )
            
            response = model.generate_content(
                contents=gemini_messages_for_api,
                generation_config=generation_config
            )
            
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise


class DefaultClient(LLMClient):
    """Default client for testing or when no specific LLM is selected."""
    
    def generate(self, messages: list, **kwargs) -> str:
        """Generate a dummy response for testing."""
        system = ""
        prompt = ""
        
        for msg in messages:
            if msg['role'] == 'system':
                system = msg['content']
            elif msg['role'] == 'user' and not prompt:
                prompt = msg['content']
        
        return f"This is a dummy response from the default client. System: '{system[:20]}...', Prompt: '{prompt[:20]}...'"