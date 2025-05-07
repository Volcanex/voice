"""
State Manager for the voice assistant.

Manages conversation history and user preferences.
"""
import logging
import time
from typing import Dict, Any, List, Optional, Set
import uuid

logger = logging.getLogger(__name__)

class Conversation:
    """
    Represents a conversation with history and metadata.
    """
    def __init__(self, max_history_length: int = 10):
        """
        Initialize a new conversation.
        
        Args:
            max_history_length: Maximum number of turns to store in history
        """
        self.id = str(uuid.uuid4())
        self.messages: List[Dict[str, str]] = []
        self.created_at = time.time()
        self.last_updated = time.time()
        self.max_history_length = max_history_length
        
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role ('user', 'assistant', or 'system')
            content: Message content
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        self.messages.append(message)
        self.last_updated = time.time()
        
        # Trim history if needed
        if len(self.messages) > self.max_history_length:
            self.messages = self.messages[-self.max_history_length:]
            
    def get_formatted_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history in the format expected by the LLM.
        
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        return [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in self.messages
        ]
        
    def clear(self) -> None:
        """
        Clear the conversation history.
        """
        self.messages = []
        self.last_updated = time.time()

class StateManager:
    """
    Manages the state of all active conversations and user preferences.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the state manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.conversations: Dict[str, Conversation] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.active_sessions: Set[str] = set()
        
        self.max_history_length = config.get("max_history_length", 10)
        self.system_prompt = config.get("system_prompt", "")
        
        logger.info("State Manager initialized")

    def create_conversation(self, session_id: str) -> str:
        """
        Create a new conversation for a session.
        
        Args:
            session_id: WebSocket session ID
            
        Returns:
            Conversation ID
        """
        conversation = Conversation(max_history_length=self.max_history_length)
        conversation_id = conversation.id
        
        self.conversations[conversation_id] = conversation
        self.active_sessions.add(session_id)
        
        logger.info(f"Created conversation {conversation_id} for session {session_id}")
        return conversation_id
        
    def add_message(
        self, conversation_id: str, role: str, content: str
    ) -> None:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            role: Message role ('user', 'assistant', or 'system')
            content: Message content
        """
        if conversation_id not in self.conversations:
            logger.warning(f"Attempted to add message to unknown conversation {conversation_id}")
            return
            
        self.conversations[conversation_id].add_message(role, content)
        
    def get_conversation_history(
        self, conversation_id: str
    ) -> List[Dict[str, str]]:
        """
        Get the formatted history for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Formatted conversation history
        """
        if conversation_id not in self.conversations:
            logger.warning(f"Attempted to get history for unknown conversation {conversation_id}")
            return []
            
        return self.conversations[conversation_id].get_formatted_history()
        
    def get_system_prompt(self) -> str:
        """
        Get the current system prompt.
        
        Returns:
            System prompt string
        """
        return self.system_prompt
        
    def set_user_preference(
        self, session_id: str, preference_key: str, preference_value: Any
    ) -> None:
        """
        Set a user preference.
        
        Args:
            session_id: WebSocket session ID
            preference_key: Preference name
            preference_value: Preference value
        """
        if session_id not in self.user_preferences:
            self.user_preferences[session_id] = {}
            
        self.user_preferences[session_id][preference_key] = preference_value
        
    def get_user_preference(
        self, session_id: str, preference_key: str, default: Any = None
    ) -> Any:
        """
        Get a user preference.
        
        Args:
            session_id: WebSocket session ID
            preference_key: Preference name
            default: Default value if preference not found
            
        Returns:
            Preference value or default
        """
        if session_id not in self.user_preferences:
            return default
            
        return self.user_preferences[session_id].get(preference_key, default)
        
    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear a conversation's history.
        
        Args:
            conversation_id: Conversation ID
        """
        if conversation_id not in self.conversations:
            logger.warning(f"Attempted to clear unknown conversation {conversation_id}")
            return
            
        self.conversations[conversation_id].clear()
        
    def remove_session(self, session_id: str) -> None:
        """
        Clean up when a session disconnects.
        
        Args:
            session_id: WebSocket session ID
        """
        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)
            
        if session_id in self.user_preferences:
            del self.user_preferences[session_id]
            
        # Note: We don't delete conversations immediately to allow reconnection
        logger.info(f"Session {session_id} removed")
        
    def cleanup_old_conversations(self, max_age_seconds: int = 3600) -> None:
        """
        Clean up conversations older than specified age.
        
        Args:
            max_age_seconds: Maximum age in seconds
        """
        current_time = time.time()
        conversations_to_remove = []
        
        for conv_id, conversation in self.conversations.items():
            if current_time - conversation.last_updated > max_age_seconds:
                conversations_to_remove.append(conv_id)
                
        for conv_id in conversations_to_remove:
            del self.conversations[conv_id]
            
        if conversations_to_remove:
            logger.info(f"Cleaned up {len(conversations_to_remove)} old conversations")