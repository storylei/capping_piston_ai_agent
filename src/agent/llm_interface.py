"""
LLM Interface Module
Supports multiple LLM backends: Ollama (local), OpenAI, Claude, Gemini, DeepSeek
"""

import json
import requests
from typing import Dict, List, Optional, Any
import os


class LLMInterface:
    """
    Unified interface for LLM interactions
    Supports: Ollama (local), OpenAI, Claude, Gemini, DeepSeek
    """
    
    # Supported backends and their default models
    SUPPORTED_BACKENDS = {
        "ollama": "llama3:latest",
        "openai": "gpt-3.5-turbo",
        "claude": "claude-3-sonnet-20240229",
        "gemini": "gemini-pro",
        "deepseek": "deepseek-chat"
    }
    
    def __init__(self, backend: str = "ollama", model: str = None, api_key: str = None):
        """
        Initialize LLM interface
        
        Args:
            backend: "ollama", "openai", "claude", "gemini", or "deepseek"
            model: Model name (uses default if not specified)
            api_key: API key (required for cloud backends)
        """
        self.backend = backend.lower()
        
        # Validate backend
        if self.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}. Supported: {list(self.SUPPORTED_BACKENDS.keys())}")
        
        # Set model
        self.model = model or self.SUPPORTED_BACKENDS[self.backend]
        
        # Set API key from parameter or environment variable
        self.api_key = api_key or self._get_env_api_key()
        
        # Initialize backend
        if self.backend == "ollama":
            self.base_url = "http://localhost:11434"
            self._check_ollama_available()
        else:
            # Cloud backends require API key
            if not self.api_key:
                print(f"⚠️  Warning: {self.backend} requires an API key but none was provided.")
    
    def _get_env_api_key(self) -> Optional[str]:
        """Get API key from environment variable based on backend"""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY"
        }
        if self.backend in env_vars:
            return os.getenv(env_vars[self.backend])
    
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
        elif self.backend == "claude":
            return self._generate_claude(messages, temperature, max_tokens, tools)
        elif self.backend == "gemini":
            return self._generate_gemini(messages, temperature, max_tokens, tools)
        elif self.backend == "deepseek":
            return self._generate_deepseek(messages, temperature, max_tokens, tools)
        else:
            return {
                'content': f"❌ Error: Unsupported backend '{self.backend}'",
                'tool_calls': None,
                'error': 'unsupported_backend'
            }
    
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
    
    def _generate_claude(self, messages: List[Dict[str, str]], 
                        temperature: float,
                        max_tokens: int,
                        tools: List[Dict] = None) -> Dict[str, Any]:
        """Generate response using Anthropic Claude API"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Claude requires system message to be separate
            system_msg = ""
            claude_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    system_msg = msg['content']
                else:
                    claude_messages.append(msg)
            
            kwargs = {
                "model": self.model,
                "messages": claude_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if system_msg:
                kwargs["system"] = system_msg
            
            response = client.messages.create(**kwargs)
            
            content = response.content[0].text if response.content else ''
            
            return {
                'content': content,
                'tool_calls': None,
                'model': self.model,
                'backend': 'claude'
            }
            
        except ImportError:
            return {
                'content': "❌ Error: Anthropic library not installed. Run: pip install anthropic",
                'tool_calls': None,
                'error': 'missing_library'
            }
        except Exception as e:
            return {
                'content': f"❌ Error: {str(e)}",
                'tool_calls': None,
                'error': str(e)
            }
    
    def _generate_gemini(self, messages: List[Dict[str, str]], 
                        temperature: float,
                        max_tokens: int,
                        tools: List[Dict] = None) -> Dict[str, Any]:
        """Generate response using Google Gemini API"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            model = genai.GenerativeModel(self.model)
            
            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    # Prepend system message to first user message
                    continue
                role = 'user' if msg['role'] == 'user' else 'model'
                gemini_messages.append({
                    'role': role,
                    'parts': [msg['content']]
                })
            
            # Get system message if exists
            system_msg = next((m['content'] for m in messages if m['role'] == 'system'), None)
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            # Start chat with history
            if len(gemini_messages) > 1:
                chat = model.start_chat(history=gemini_messages[:-1])
                last_msg = gemini_messages[-1]['parts'][0]
                if system_msg:
                    last_msg = f"[System: {system_msg}]\n\n{last_msg}"
                response = chat.send_message(last_msg, generation_config=generation_config)
            else:
                prompt = gemini_messages[0]['parts'][0] if gemini_messages else ""
                if system_msg:
                    prompt = f"[System: {system_msg}]\n\n{prompt}"
                response = model.generate_content(prompt, generation_config=generation_config)
            
            content = response.text if response.text else ''
            
            return {
                'content': content,
                'tool_calls': None,
                'model': self.model,
                'backend': 'gemini'
            }
            
        except ImportError:
            return {
                'content': "❌ Error: Google Generative AI library not installed. Run: pip install google-generativeai",
                'tool_calls': None,
                'error': 'missing_library'
            }
        except Exception as e:
            return {
                'content': f"❌ Error: {str(e)}",
                'tool_calls': None,
                'error': str(e)
            }
    
    def _generate_deepseek(self, messages: List[Dict[str, str]], 
                          temperature: float,
                          max_tokens: int,
                          tools: List[Dict] = None) -> Dict[str, Any]:
        """Generate response using DeepSeek API (OpenAI-compatible)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            return {
                'content': content,
                'tool_calls': None,
                'model': self.model,
                'backend': 'deepseek'
            }
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"DeepSeek API error: {e.response.status_code}"
            if e.response.status_code == 401:
                error_msg = "❌ Error: Invalid DeepSeek API key"
            return {
                'content': error_msg,
                'tool_calls': None,
                'error': str(e)
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
        elif self.backend == "claude":
            yield from self._stream_claude(messages, temperature)
        elif self.backend == "gemini":
            yield from self._stream_gemini(messages, temperature)
        elif self.backend == "deepseek":
            yield from self._stream_deepseek(messages, temperature)
        else:
            yield f"❌ Error: Unsupported backend '{self.backend}'"
    
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
    
    def _stream_claude(self, messages: List[Dict[str, str]], temperature: float):
        """Stream from Claude"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Claude requires system message to be separate
            system_msg = ""
            claude_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    system_msg = msg['content']
                else:
                    claude_messages.append(msg)
            
            kwargs = {
                "model": self.model,
                "messages": claude_messages,
                "temperature": temperature,
                "max_tokens": 2000
            }
            
            if system_msg:
                kwargs["system"] = system_msg
            
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            yield f"❌ Error: {str(e)}"
    
    def _stream_gemini(self, messages: List[Dict[str, str]], temperature: float):
        """Stream from Gemini"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            model = genai.GenerativeModel(self.model)
            
            # Convert messages to prompt
            prompt_parts = []
            for msg in messages:
                if msg['role'] == 'system':
                    prompt_parts.append(f"[System: {msg['content']}]")
                else:
                    prompt_parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
            
            prompt = "\n".join(prompt_parts)
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature
            )
            
            response = model.generate_content(prompt, generation_config=generation_config, stream=True)
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            yield f"❌ Error: {str(e)}"
    
    def _stream_deepseek(self, messages: List[Dict[str, str]], temperature: float):
        """Stream from DeepSeek (OpenAI-compatible)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": True
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
                timeout=60
            )
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]
                        if line.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(line)
                            delta = chunk['choices'][0]['delta']
                            if 'content' in delta:
                                yield delta['content']
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            yield f"❌ Error: {str(e)}"