"""Tests for multi-agent system."""

import pytest
from src.mmllm.multi_agent import MultiAgentCoordinator, MultiAgentState, AgentPhase
from src.mmllm.utils import EpisodeLoader
from src.mmllm.actions import ActionValidator
from src.mmllm.vision import ImageProcessor
import logging

logger = logging.getLogger(__name__)

class TestMultiAgent:
    """Test multi-agent coordination."""
    
    def test_multi_agent_coordinator_initialization(self):
        """Test that coordinator initializes properly."""
        coordinator = MultiAgentCoordinator()
        assert coordinator.graph is not None
    
    def test_episode_loader_mock_state(self):
        """Test episode loader creates valid mock state."""
        loader = EpisodeLoader()
        state = loader._create_mock_state()
        
        assert isinstance(state, dict)
        assert "goal" in state
        assert "current_image" in state
        assert "ui_annotations" in state
        assert state["current_phase"] == AgentPhase.PLANNING
    
    def test_action_validator(self):
        """Test action validator basic functionality."""
        validator = ActionValidator()
        # This would need mock action plans and UI annotations
        assert validator is not None
    
    def test_image_processor(self):
        """Test image processor basic functionality."""
        processor = ImageProcessor()
        assert processor.max_image_size == 1024


class TestActionSchemas:
    """Test action schema validation."""
    
    def test_tap_action_validation(self):
        """Test tap action coordinate validation."""
        from src.mmllm.actions.action_schemas import TapAction
        from src.mmllm.android_in_the_wild.action_type import ActionType
        
        # Valid tap action
        tap = TapAction(
            action_type=ActionType.DUAL_POINT,
            coordinates=[0.5, 0.5],
            reasoning="Test tap",
            confidence=0.8
        )
        assert tap.coordinates == [0.5, 0.5]
        
        # Invalid coordinates should raise validation error
        with pytest.raises(ValueError):
            TapAction(
                action_type=ActionType.DUAL_POINT,
                coordinates=[1.5, 0.5],  # Out of bounds
                reasoning="Test tap",
                confidence=0.8
            )


class TestVisionProcessing:
    """Test vision processing components."""
    
    def test_annotation_parser(self):
        """Test UI annotation parsing."""
        from src.mmllm.vision import AnnotationParser
        parser = AnnotationParser()
        assert parser is not None


# Integration test function that can be run manually
def test_multi_agent_integration():
    """Integration test with mock data."""
    logger.info("Running integration test...")
    
    # Load mock state
    loader = EpisodeLoader()
    initial_state = loader._create_mock_state()
    
    # Initialize coordinator
    coordinator = MultiAgentCoordinator()
    
    # Run the system
    try:
        final_state = coordinator.run(initial_state)
        logger.info(f"Test completed. Final phase: {final_state.get('current_phase')}")
        logger.info(f"Steps: {final_state.get('current_step')}")
        return True
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


if __name__ == "__main__":
    # Run integration test when script is executed directly
    test_multi_agent_integration()
