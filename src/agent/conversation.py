"""
Conversation Manager Module
Manages chat history and context for the AI agent
"""

from typing import List, Dict, Any
from datetime import datetime


class ConversationManager:
    """Manages conversation state and history"""
    
    def __init__(self, max_history: int = 20):
        """
        Initialize conversation manager
        
        Args:
            max_history: Maximum number of messages to keep in history
        """
        self.max_history = max_history
        self.messages: List[Dict[str, Any]] = []
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the AI agent"""
        return """You are a Statistical Analysis AI Agent specialized in analyzing industrial sensor datasets with OK/KO labels (e.g., NASA C-MAPSS turbofan engine degradation data).

Your capabilities include:
1. Statistical Analysis: Calculate mean, median, mode, standard deviation, variance for sensor features
2. Feature Importance: Identify which sensors best discriminate between OK (healthy) and KO (degrading) states
3. Data Visualization:
   - Histograms (sensor value distribution)
   - Box plots (show quartiles and outliers)
   - Violin plots (show distribution shape)
   - KDE plots (smooth density curves)
   - Time series plots (sensor readings over time/cycles)
   - FFT/Frequency spectrum plots (frequency domain analysis)
4. Multi-feature Comparison: Compare multiple sensors side by side
5. Group Filtering: Show data for OK samples only or KO samples only

Example queries:
- 'Show mean and std for sensor_2'
- 'Plot histogram of sensor_11'
- 'Show time series for KO samples of sensor_7'
- 'Plot FFT for sensor_4'
- 'Get feature importance ranking'

When the user asks for analysis:
1. Understand their intent
2. Use appropriate visualization for the data type
3. Present clear, informative results with OK/KO comparisons"""
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """
        Add a message to conversation history
        
        Args:
            role: 'user', 'assistant', or 'system'
            content: Message content
            metadata: Additional metadata (tool calls, etc.)
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            message['metadata'] = metadata
        
        self.messages.append(message)
        
        # Trim history if too long (keep system message)
        if len(self.messages) > self.max_history + 1:
            # Keep first message (system) and most recent messages
            self.messages = [self.messages[0]] + self.messages[-(self.max_history):]
    
    def get_messages_for_llm(self, include_system: bool = True) -> List[Dict[str, str]]:
        """
        Get messages formatted for LLM API
        
        Args:
            include_system: Whether to include system prompt
            
        Returns:
            List of message dicts with 'role' and 'content'
        """
        messages = []
        
        if include_system:
            messages.append({
                'role': 'system',
                'content': self.system_prompt
            })
        
        # Add conversation history (excluding metadata for LLM)
        for msg in self.messages:
            if msg['role'] != 'system':  # Skip system messages from history
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        return messages
    
    # [Reserved] Not currently used - for retrieving complete conversation history with metadata
    def get_full_history(self) -> List[Dict[str, Any]]:
        """Get complete conversation history with metadata"""
        return self.messages.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.messages = []
    
    # [Reserved] Not currently used - for future AI mode switching
    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt"""
        self.system_prompt = new_prompt
    
    def add_context(self, context: str):
        """Add context information to system prompt"""
        self.system_prompt += f"\n\nCurrent Context:\n{context}"
    
    # [Reserved] Not currently used - for retrieving recent N messages
    def get_last_n_messages(self, n: int) -> List[Dict[str, Any]]:
        """Get last n messages"""
        return self.messages[-n:] if len(self.messages) >= n else self.messages.copy()
