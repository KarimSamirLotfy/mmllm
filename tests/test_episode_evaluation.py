"""Integration tests for episode evaluation framework."""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.mmllm.evaluation.episode_runner import EpisodeRunner
from src.mmllm.evaluation.action_comparator import ActionComparator
from src.mmllm.evaluation.evaluation_reporter import EvaluationReporter
from src.mmllm.multi_agent.enhanced_planning_agent import EnhancedPlanningAgent
from src.mmllm.multi_agent.state import (
    StepEvaluationResult, 
    EpisodeEvaluationState, 
    PlanningOutput,
    AgentPhase
)
from src.mmllm.android_in_the_wild.action_type import ActionType
from src.mmllm.utils.episode_loader import EpisodeLoader

logger = logging.getLogger(__name__)


class TestEpisodeEvaluation:
    """Test episode evaluation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.episode_runner = EpisodeRunner()
        self.action_comparator = ActionComparator()
        self.evaluation_reporter = EvaluationReporter()
        self.enhanced_planning_agent = EnhancedPlanningAgent()
        self.episode_loader = EpisodeLoader()
    
    def test_action_comparator_dual_point_exact_match(self):
        """Test action comparator with exact coordinate match."""
        agent_action = {
            "action_type": ActionType.DUAL_POINT.value,
            "coordinates": [0.5, 0.5]
        }
        ground_truth_action = {
            "action_type": ActionType.DUAL_POINT.value,
            "coordinates": [0.5, 0.5]
        }
        
        result = self.action_comparator.compare_actions(agent_action, ground_truth_action, 1)
        
        assert result.action_match is True
        assert result.action_type_match is True
        assert result.evaluation_score == 1.0
        assert result.coordinate_distance == 0.0
    
    def test_action_comparator_dual_point_close_match(self):
        """Test action comparator with close coordinate match."""
        agent_action = {
            "action_type": ActionType.DUAL_POINT.value,
            "coordinates": [0.5, 0.5]
        }
        ground_truth_action = {
            "action_type": ActionType.DUAL_POINT.value,
            "coordinates": [0.51, 0.51]
        }
        
        result = self.action_comparator.compare_actions(agent_action, ground_truth_action, 1)
        
        assert result.action_type_match is True
        assert result.coordinate_distance is not None
        assert result.coordinate_distance < 0.14  # Within threshold
        assert result.evaluation_score == 1.0  # Should be perfect match within threshold
    
    def test_action_comparator_dual_point_far_match(self):
        """Test action comparator with far coordinate match."""
        agent_action = {
            "action_type": ActionType.DUAL_POINT.value,
            "coordinates": [0.1, 0.1]
        }
        ground_truth_action = {
            "action_type": ActionType.DUAL_POINT.value,
            "coordinates": [0.9, 0.9]
        }
        
        result = self.action_comparator.compare_actions(agent_action, ground_truth_action, 1)
        
        assert result.action_type_match is True
        assert result.action_match is False
        assert result.coordinate_distance is not None
        assert result.coordinate_distance > 0.14  # Outside threshold
        assert 0.0 < result.evaluation_score < 1.0  # Partial score
    
    def test_action_comparator_type_action_exact_match(self):
        """Test action comparator with exact text match."""
        agent_action = {
            "action_type": ActionType.TYPE.value,
            "text": "hello world"
        }
        ground_truth_action = {
            "action_type": ActionType.TYPE.value,
            "text": "hello world"
        }
        
        result = self.action_comparator.compare_actions(agent_action, ground_truth_action, 1)
        
        assert result.action_match is True
        assert result.action_type_match is True
        assert result.text_match is True
        assert result.evaluation_score == 1.0
    
    def test_action_comparator_type_action_different_text(self):
        """Test action comparator with different text."""
        agent_action = {
            "action_type": ActionType.TYPE.value,
            "text": "hello world"
        }
        ground_truth_action = {
            "action_type": ActionType.TYPE.value,
            "text": "goodbye world"
        }
        
        result = self.action_comparator.compare_actions(agent_action, ground_truth_action, 1)
        
        assert result.action_type_match is True
        assert result.text_match is False
        assert 0.0 < result.evaluation_score < 1.0  # Partial score based on similarity
    
    def test_action_comparator_different_action_types(self):
        """Test action comparator with different action types."""
        agent_action = {
            "action_type": ActionType.TYPE.value,
            "text": "hello"
        }
        ground_truth_action = {
            "action_type": ActionType.DUAL_POINT.value,
            "coordinates": [0.5, 0.5]
        }
        
        result = self.action_comparator.compare_actions(agent_action, ground_truth_action, 1)
        
        assert result.action_type_match is False
        assert result.action_match is False
        assert result.evaluation_score == 0.1  # Small score for attempting action
    
    def test_enhanced_planning_agent_initialization(self):
        """Test enhanced planning agent initializes correctly."""
        agent = EnhancedPlanningAgent()
        assert agent.model is not None
        assert agent.max_historical_images == 3
    
    @patch('src.mmllm.multi_agent.planning_agent.get_model')
    def test_enhanced_planning_agent_with_history(self, mock_get_model):
        """Test enhanced planning agent with historical context."""
        # Mock model
        mock_model = Mock()
        mock_structured_model = Mock()
        mock_model.with_structured_output.return_value = mock_structured_model
        mock_structured_model.invoke.return_value = PlanningOutput(
            action_type=ActionType.DUAL_POINT,
            coordinates=[0.5, 0.5],
            reasoning="Tap center based on historical analysis",
            confidence=0.8,
            historical_context_used=["3_previous_screens", "2_previous_actions"]
        )
        mock_get_model.return_value = mock_model
        
        agent = EnhancedPlanningAgent()
        
        # Create test state with historical context
        state: EpisodeEvaluationState = {
            "goal": "Find and tap the button",
            "current_step": 2,
            "max_steps": 5,
            "current_phase": AgentPhase.PLANNING,
            "current_image": "base64_current_image",
            "ui_annotations": [],
            "episode_images": ["base64_img1", "base64_img2", "base64_current_image"],
            "episode_id": "test_episode",
            "episode_length": 3,
            "current_ground_truth": {"action_type": ActionType.DUAL_POINT.value},
            "ground_truth_actions": [],
            "messages": [],
            "planning_output": None,
            "execution_output": None,
            "reflection_output": None,
            "action_history": [],
            "execution_history": [],
            "reflection_history": [],
            "error_count": 0,
            "last_error": None,
            "step_evaluations": None,
            "final_result": None
        }
        
        result = agent.plan_action_with_history(state)
        
        assert result.goto == "execution_agent"
        assert "planning_output" in result.update
        assert result.update["planning_output"].historical_context_used is not None
        
        # Verify model was called with multimodal content
        mock_structured_model.invoke.assert_called_once()
        call_args = mock_structured_model.invoke.call_args[0][0]  # messages
        assert len(call_args) == 2  # SystemMessage and HumanMessage
        assert any("historical context" in str(msg).lower() for msg in call_args)
    
    def test_episode_loader_load_episode_with_history(self):
        """Test episode loader with historical context."""
        # Create mock episode examples
        mock_examples = []
        for i in range(3):
            mock_example = Mock()
            mock_example.features.feature = {
                'episode_id': Mock(bytes_list=Mock(value=[b'test_episode_123'])),
                'action_type': Mock(int64_list=Mock(value=[ActionType.DUAL_POINT.value])),
                'touch_point_x': Mock(float_list=Mock(value=[0.5])),
                'touch_point_y': Mock(float_list=Mock(value=[0.5]))
            }
            mock_examples.append(mock_example)
        
        # Mock the image processor and annotation parser
        with patch.object(self.episode_loader.image_processor, 'decode_aitw_image') as mock_decode, \
             patch.object(self.episode_loader.image_processor, 'encode_image_for_model') as mock_encode, \
             patch.object(self.episode_loader.annotation_parser, 'extract_episode_info') as mock_extract_info, \
             patch.object(self.episode_loader.annotation_parser, 'parse_ui_annotations') as mock_parse_ui:
            
            mock_decode.return_value = "mock_image_array"
            mock_encode.return_value = f"base64_image"
            mock_extract_info.return_value = {
                "episode_id": "test_episode_123",
                "goal_info": "Test goal"
            }
            mock_parse_ui.return_value = []
            
            result = self.episode_loader.load_episode_with_history(mock_examples)
            
            assert result["episode_id"] == "test_episode_123"
            assert result["goal"] == "Test goal"
            assert result["episode_length"] == 3
            assert len(result["episode_images"]) == 3
            assert len(result["ground_truth_actions"]) == 3
    
    def test_episode_loader_create_evaluation_state(self):
        """Test creating evaluation state for a specific step."""
        episode_data = {
            "episode_id": "test_episode",
            "goal": "Test goal",
            "episode_length": 3,
            "episode_images": ["img1", "img2", "img3"],
            "ground_truth_actions": [
                {"action_type": ActionType.DUAL_POINT.value},
                {"action_type": ActionType.TYPE.value},
                {"action_type": ActionType.PRESS_BACK.value}
            ],
            "ui_annotations_list": [[], [], []]
        }
        
        state = self.episode_loader.create_evaluation_state_for_step(episode_data, 1)
        
        assert state["goal"] == "Test goal"
        assert state["current_step"] == 1
        assert state["current_image"] == "img2"
        assert len(state["episode_images"]) == 2  # Historical context: img1, img2
        assert state["current_ground_truth"]["action_type"] == ActionType.TYPE.value
    
    def test_evaluation_reporter_generate_episode_report(self):
        """Test evaluation reporter generates correct episode report."""
        step_evaluations = [
            StepEvaluationResult(
                step_number=0,
                agent_action={"action_type": ActionType.DUAL_POINT.value},
                ground_truth_action={"action_type": ActionType.DUAL_POINT.value},
                action_match=True,
                action_type_match=True,
                evaluation_score=1.0
            ),
            StepEvaluationResult(
                step_number=1,
                agent_action={"action_type": ActionType.TYPE.value},
                ground_truth_action={"action_type": ActionType.TYPE.value},
                action_match=False,
                action_type_match=True,
                evaluation_score=0.7
            )
        ]
        
        episode_result = {
            "episode_id": "test_episode",
            "total_steps": 2,
            "step_evaluations": step_evaluations,
            "overall_success_rate": 0.85
        }
        
        report = self.evaluation_reporter.generate_episode_report(episode_result, save_to_file=False)
        
        assert "episode_summary" in report
        assert report["episode_summary"]["episode_id"] == "test_episode"
        assert report["episode_summary"]["overall_success_rate"] == 0.85
        assert report["episode_summary"]["completed_steps"] == 1
        assert report["episode_summary"]["failed_steps"] == 1
        
        assert "performance_metrics" in report
        assert "step_by_step_analysis" in report
        assert len(report["step_by_step_analysis"]) == 2
    
    def test_evaluation_reporter_generate_markdown(self):
        """Test evaluation reporter generates markdown format."""
        report = {
            "report_metadata": {"timestamp": "2024-01-01T00:00:00"},
            "episode_summary": {
                "episode_id": "test_episode",
                "total_steps": 2,
                "overall_success_rate": 0.85,
                "completed_steps": 1,
                "failed_steps": 1
            },
            "performance_metrics": {
                "action_type_accuracy": 1.0,
                "coordinate_accuracy": 0.5,
                "avg_coordinate_distance": 0.05
            }
        }
        
        markdown = self.evaluation_reporter.generate_markdown_report(report)
        
        assert "# Episode Evaluation Report: test_episode" in markdown
        assert "Success Rate**: 85.0%" in markdown
        assert "Total Steps**: 2" in markdown
        assert "Action Type Accuracy**: 100.0%" in markdown
    
    @patch('src.mmllm.evaluation.episode_runner.MultiAgentCoordinator')
    def test_episode_runner_evaluate_step(self, mock_coordinator_class):
        """Test episode runner step evaluation."""
        # Mock coordinator
        mock_coordinator = Mock()
        mock_coordinator_class.return_value = mock_coordinator
        
        # Mock coordinator result
        mock_planning_output = PlanningOutput(
            action_type=ActionType.DUAL_POINT,
            coordinates=[0.5, 0.5],
            reasoning="Test action",
            confidence=0.8
        )
        
        mock_execution_output = Mock()
        mock_execution_output.action_executed = True
        
        mock_coordinator.run.return_value = {
            "planning_output": mock_planning_output,
            "execution_output": mock_execution_output
        }
        
        runner = EpisodeRunner()
        
        episode_data = {
            "episode_id": "test_episode",
            "goal": "Test goal",
            "episode_length": 1,
            "episode_images": ["base64_image"],
            "ground_truth_actions": [
                {"action_type": ActionType.DUAL_POINT.value, "coordinates": [0.5, 0.5]}
            ],
            "ui_annotations_list": [[]]
        }
        
        result = runner._evaluate_step(episode_data, 0)
        
        assert isinstance(result, StepEvaluationResult)
        assert result.step_number == 0
        assert result.action_type_match is True
        mock_coordinator.run.assert_called_once()
    
    def test_format_agent_action_for_aitw(self):
        """Test converting PlanningOutput to AiTW format."""
        planning_output = PlanningOutput(
            action_type=ActionType.DUAL_POINT,
            coordinates=[0.5, 0.5],
            lift_coordinates=[0.6, 0.6],
            reasoning="Test action",
            confidence=0.8
        )
        
        aitw_action = self.action_comparator.format_agent_action_for_aitw(planning_output)
        
        assert aitw_action["action_type"] == ActionType.DUAL_POINT.value
        assert aitw_action["coordinates"] == [0.5, 0.5]
        assert aitw_action["lift_coordinates"] == [0.6, 0.6]
    
    def test_episode_runner_handles_errors_gracefully(self):
        """Test episode runner handles errors gracefully."""
        runner = EpisodeRunner()
        
        # Test with empty episode
        result = runner.run_episode_evaluation([], max_steps=1)
        
        assert "error" in result
        assert result["total_steps"] == 0
        assert result["overall_success_rate"] == 0.0


def create_mock_episode():
    """Create mock episode for testing."""
    mock_examples = []
    for i in range(2):
        mock_example = Mock()
        mock_example.features.feature = {
            'episode_id': Mock(bytes_list=Mock(value=[b'mock_episode'])),
            'action_type': Mock(int64_list=Mock(value=[ActionType.DUAL_POINT.value])),
            'touch_point_x': Mock(float_list=Mock(value=[0.5])),
            'touch_point_y': Mock(float_list=Mock(value=[0.5]))
        }
        mock_examples.append(mock_example)
    return mock_examples


def create_test_state_with_history() -> EpisodeEvaluationState:
    """Create test state with historical context."""
    return {
        "goal": "Test goal",
        "current_step": 1,
        "max_steps": 3,
        "current_phase": AgentPhase.PLANNING,
        "current_image": "base64_current",
        "ui_annotations": [],
        "episode_images": ["base64_prev", "base64_current"],
        "episode_id": "test_episode",
        "episode_length": 2,
        "current_ground_truth": {"action_type": ActionType.DUAL_POINT.value},
        "ground_truth_actions": [],
        "messages": [],
        "planning_output": None,
        "execution_output": None,
        "reflection_output": None,
        "action_history": [],
        "execution_history": [],
        "reflection_history": [],
        "error_count": 0,
        "last_error": None,
        "step_evaluations": None,
        "final_result": None
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
