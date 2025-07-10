"""State management utilities for agent persistence."""

from typing import Dict, Any, Optional
import json
import os
from ..multi_agent.state import MultiAgentState


class StateManager:
    """Manages agent state persistence and recovery."""
    
    def __init__(self, state_dir: str = "./agent_states"):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
    
    def save_state(self, state: MultiAgentState, episode_id: str) -> str:
        """Save agent state to disk."""
        filepath = os.path.join(self.state_dir, f"{episode_id}_state.json")
        
        # Convert state to serializable format
        serializable_state = self._make_serializable(state)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_state, f, indent=2)
        
        return filepath
    
    def load_state(self, episode_id: str) -> Optional[MultiAgentState]:
        """Load agent state from disk."""
        filepath = os.path.join(self.state_dir, f"{episode_id}_state.json")
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            serializable_state = json.load(f)
        
        return self._deserialize_state(serializable_state)
    
    def _make_serializable(self, state: MultiAgentState) -> Dict[str, Any]:
        """Convert state to JSON-serializable format."""
        serializable = {}
        
        for key, value in state.items():
            if hasattr(value, 'model_dump'):  # Pydantic models
                serializable[key] = value.model_dump()
            elif hasattr(value, '__dict__'):  # Other objects
                serializable[key] = str(value)
            else:
                serializable[key] = value
        
        return serializable
    
    def _deserialize_state(self, data: Dict[str, Any]) -> MultiAgentState:
        """Convert serialized data back to MultiAgentState."""
        # This is a simplified deserialization
        # In production, you'd want proper type reconstruction
        return data
    
    def cleanup_old_states(self, max_age_hours: int = 24):
        """Remove old state files."""
        import time
        current_time = time.time()
        
        for filename in os.listdir(self.state_dir):
            if filename.endswith('_state.json'):
                filepath = os.path.join(self.state_dir, filename)
                file_age = current_time - os.path.getctime(filepath)
                
                if file_age > max_age_hours * 3600:  # Convert hours to seconds
                    os.remove(filepath)
