import os
import requests
import json
from typing import Generator

class RemoteModel:
    """Base class for remote model families"""
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning_mode = False
    
    def set_reasoning_mode(self, reasoning: bool):
        """Set reasoning mode which affects model selection"""
        self.reasoning_mode = reasoning
    
    def load(self):
        """Initialize connection to remote model API"""
        pass
    
    def generate(self, prompt, params=None):
        """Generate text from the remote model"""
        pass
    
    def generate_stream(self, prompt, params=None):
        """Generate text from the remote model with streaming"""
        pass

class DeepSeek(RemoteModel):
    """DeepSeek model family"""
    
    def __init__(self, base_url=None):
        # Get API key from environment variable
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
            
        super().__init__(api_key, base_url)
        
        # Set default base URL if not provided
        if not self.base_url:
            self.base_url = "https://api.deepseek.com/v1"
    
    @property
    def model_name(self):
        """Get model name based on reasoning mode"""
        if self.reasoning_mode:
            return "deepseek-reasoner"  # R1
        else:
            return "deepseek-chat"      # V3
    
    def generate_stream(self, prompt, params=None):
        """Generate with streaming for DeepSeek API"""
        if not params:
            params = {}
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        
        # Add additional parameters
        for key, value in params.items():
            if key not in data:
                data[key] = value
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: ') and line != 'data: [DONE]':
                    try:
                        chunk = json.loads(line[6:])
                        if chunk.get('choices') and chunk['choices'][0].get('delta'):
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

class OpenAI(RemoteModel):
    """OpenAI model family"""
    
    def __init__(self, base_url=None):
        # Get API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        super().__init__(api_key, base_url)
        
        # Set default base URL if not provided
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"
    
    @property
    def model_name(self):
        """Get model name based on reasoning mode"""
        if self.reasoning_mode:
            return "o1-mini"  # Reasoning model
        else:
            return "gpt-4o"   # Non-reasoning model
    
    def generate_stream(self, prompt, params=None):
        """Generate with streaming for OpenAI API"""
        if not params:
            params = {}
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        
        # Add additional parameters
        for key, value in params.items():
            if key not in data:
                data[key] = value
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: ') and line != 'data: [DONE]':
                    try:
                        chunk = json.loads(line[6:])
                        if chunk.get('choices') and chunk['choices'][0].get('delta'):
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

class Claude(RemoteModel):
    """Claude model family"""
    
    def __init__(self, base_url=None):
        # Get API key from environment variable
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
        super().__init__(api_key, base_url)
        
        # Set default base URL if not provided
        if not self.base_url:
            self.base_url = "https://api.anthropic.com/v1"
    
    @property
    def model_name(self):
        """Get model name based on reasoning mode"""
        if self.reasoning_mode:
            return "claude-3-5-sonnet-20241022"  # Sonnet 4 (reasoning)
        else:
            return "claude-3-5-haiku-20241022"   # Haiku (non-reasoning)
    
    def generate_stream(self, prompt, params=None):
        """Generate with streaming for Claude API"""
        if not params:
            params = {}
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 4096
        }
        
        # Add additional parameters
        for key, value in params.items():
            if key not in data:
                data[key] = value
        
        response = requests.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=data,
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        chunk = json.loads(line[6:])
                        if chunk.get('type') == 'content_block_delta':
                            content = chunk.get('delta', {}).get('text', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue 