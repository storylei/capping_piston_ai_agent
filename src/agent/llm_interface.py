"""
LLM Interface Module
Supports both local LLM (Ollama) and OpenAI API for flexible deployment
"""

import json
import requests
from typing import Dict, List, Optional, Any
import os


class LLMInterface:
    """
    Unified interface for LLM interactions
    Supports: Ollama (local LLAMA3), OpenAI API
    """
    
    def __init__(self, backend: str = "ollama", model: str = None, api_key: str = None):
        """
        Initialize LLM interface
        
        Args:
            backend: "ollama" for local LLAMA3, "openai" for OpenAI API
            model: Model name (e.g., "llama3", "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (required if backend="openai")
        """
        self.backend = backend.lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if self.backend == "ollama":
            self.model = model or "llama3:latest"
            self.base_url = "http://localhost:11434"
            self._check_ollama_available()
        elif self.backend == "openai":
            self.model = model or "gpt-3.5-turbo"
            if not self.api_key:
                raise ValueError("OpenAI API key required for 'openai' backend")
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _check_ollama_available(self):
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                if self.model not in models:
                    print(f"⚠️  Warning: Model '{self.model}' not found in Ollama.")
                    print(f"   Available models: {models}")
                    print(f"   To install: ollama pull {self.model}")
            return True
        except requests.exceptions.ConnectionError:
            print("⚠️  Warning: Ollama service not running.")
            print("   Please start Ollama or switch to 'openai' backend")
            print("   Install: https://ollama.ai/download")
            return False
        except Exception as e:
            print(f"⚠️  Error checking Ollama: {str(e)}")
            return False
    
    def generate(self, messages: List[Dict[str, str]], 
                temperature: float = 0.7,
                max_tokens: int = 2000,
                tools: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate response from LLM
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: List of available tools for function calling
            
        Returns:
            Dictionary with 'content' and optional 'tool_calls'
        """
        if self.backend == "ollama":
            return self._generate_ollama(messages, temperature, max_tokens, tools)
        elif self.backend == "openai":
            return self._generate_openai(messages, temperature, max_tokens, tools)
    
    def _generate_ollama(self, messages: List[Dict[str, str]], 
                        temperature: float,
                        max_tokens: int,
                        tools: List[Dict] = None) -> Dict[str, Any]:
        """Generate response using Ollama API"""
        try:
            # Ollama uses a simpler format
            # Convert messages to prompt format
            prompt = self._messages_to_prompt(messages, tools)
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get('response', '')
            
            # Parse for tool calls if tools are provided
            tool_calls = None
            if tools:
                tool_calls = self._parse_tool_calls(content)
            
            return {
                'content': content,
                'tool_calls': tool_calls,
                'model': self.model,
                'backend': 'ollama'
            }
            
        except requests.exceptions.ConnectionError:
            return {
                'content': "❌ Error: Cannot connect to Ollama. Please start Ollama service.",
                'tool_calls': None,
                'error': 'connection_failed'
            }
        except Exception as e:
            return {
                'content': f"❌ Error: {str(e)}",
                'tool_calls': None,
                'error': str(e)
            }
    
    def _generate_openai(self, messages: List[Dict[str, str]], 
                        temperature: float,
                        max_tokens: int,
                        tools: List[Dict] = None) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        try:
            import openai
            openai.api_key = self.api_key
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = openai.chat.completions.create(**kwargs)
            
            message = response.choices[0].message
            
            return {
                'content': message.content or '',
                'tool_calls': message.tool_calls if hasattr(message, 'tool_calls') else None,
                'model': self.model,
                'backend': 'openai'
            }
            
        except ImportError:
            return {
                'content': "❌ Error: OpenAI library not installed. Run: pip install openai",
                'tool_calls': None,
                'error': 'missing_library'
            }
        except Exception as e:
            return {
                'content': f"❌ Error: {str(e)}",
                'tool_calls': None,
                'error': str(e)
            }
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> str:
        """Convert messages to a single prompt for Ollama"""
        prompt_parts = []
        
        # Add system message if present
        system_msgs = [m for m in messages if m['role'] == 'system']
        if system_msgs:
            prompt_parts.append(f"System: {system_msgs[0]['content']}\n")
        
        # Add tool descriptions if provided
        if tools:
            prompt_parts.append("\nAvailable Tools:")
            for tool in tools:
                func = tool.get('function', {})
                prompt_parts.append(f"- {func.get('name')}: {func.get('description')}")
            prompt_parts.append("\nTo use a tool, respond with: TOOL_CALL: {\"name\": \"tool_name\", \"arguments\": {...}}\n")
        
        # Add conversation history
        for msg in messages:
            if msg['role'] == 'system':
                continue
            role = msg['role'].capitalize()
            prompt_parts.append(f"\n{role}: {msg['content']}")
        
        prompt_parts.append("\n\nAssistant:")
        
        return "\n".join(prompt_parts)
    
    def _parse_tool_calls(self, content: str) -> Optional[List[Dict]]:
        """Parse tool calls from Ollama response"""
        if "TOOL_CALL:" not in content:
            return None
        
        try:
            # Extract JSON after TOOL_CALL:
            parts = content.split("TOOL_CALL:")
            if len(parts) < 2:
                return None
            
            json_str = parts[1].strip()
            # Find the JSON object
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start == -1 or end == 0:
                return None
            
            tool_call_data = json.loads(json_str[start:end])
            
            return [{
                'type': 'function',
                'function': {
                    'name': tool_call_data.get('name'),
                    'arguments': json.dumps(tool_call_data.get('arguments', {}))
                }
            }]
        except Exception:
            return None
    
    def stream_generate(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7):
        """
        Stream response from LLM (for real-time chat display)
        
        Yields:
            Chunks of generated text
        """
        if self.backend == "ollama":
            yield from self._stream_ollama(messages, temperature)
        elif self.backend == "openai":
            yield from self._stream_openai(messages, temperature)
    
    def _stream_ollama(self, messages: List[Dict[str, str]], temperature: float):
        """Stream from Ollama"""
        try:
            prompt = self._messages_to_prompt(messages)
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": temperature}
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=60
            )
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        yield chunk['response']
                        
        except Exception as e:
            yield f"❌ Error: {str(e)}"
    
    def _stream_openai(self, messages: List[Dict[str, str]], temperature: float):
        """Stream from OpenAI"""
        try:
            import openai
            openai.api_key = self.api_key
            
            stream = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"❌ Error: {str(e)}"
