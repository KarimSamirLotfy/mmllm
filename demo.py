#!/usr/bin/env python3
"""
Demo script for the multi-agent Android in the Wild system.
"""

import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.mmllm.multi_agent import MultiAgentCoordinator
from src.mmllm.utils import EpisodeLoader
from src.mmllm.multi_agent.state import AgentPhase

logger = logging.getLogger(__name__)


def run_demo():
    """Run the multi-agent demo."""
    logger.info("🤖 Multi-Agent Android in the Wild Demo")
    logger.info("=" * 50)
    
    # Initialize components
    logger.info("Initializing components...")
    episode_loader = EpisodeLoader()
    coordinator = MultiAgentCoordinator()
    
    # Load test episode
    logger.info("Loading test episode...")
    try:
        initial_state = episode_loader.get_sample_episode_state('google_apps')
        logger.info(f"✅ Loaded real episode: {initial_state['episode_id']}")
    except Exception as e:
        logger.warning(f"⚠️  Could not load real episode ({e})")
        logger.info("Using mock episode instead...")
        initial_state = episode_loader._create_mock_state()
    
    logger.info(f"🎯 Goal: {initial_state['goal']}")
    logger.info(f"📱 Max steps: {initial_state['max_steps']}")
    logger.info(f"🎨 UI elements: {len(initial_state['ui_annotations'])}")
    
    # Run multi-agent system
    logger.info("🚀 Starting Multi-Agent Execution")
    logger.info("-" * 30)
    
    try:
        final_state = coordinator.run(initial_state)
        
        # Print results
        logger.info("📊 Execution Results")
        logger.info("-" * 20)
        logger.info(f"Final phase: {final_state.get('current_phase', 'unknown')}")
        logger.info(f"Steps completed: {final_state.get('current_step', 0)}/{final_state.get('max_steps', 0)}")
        logger.info(f"Errors encountered: {final_state.get('error_count', 0)}")
        
        # Show agent outputs
        if final_state.get('planning_output'):
            planning = final_state['planning_output']
            logger.debug(f"🧠 Last planning: {planning.action_type.name} (confidence: {planning.confidence:.2f})")
            logger.debug(f"   Reasoning: {planning.reasoning}")
        
        if final_state.get('execution_output'):
            execution = final_state['execution_output']
            logger.debug(f"⚡ Last execution: {'Success' if execution.action_executed else 'Failed'}")
            logger.debug(f"   Details: {execution.execution_details}")
        
        if final_state.get('reflection_output'):
            reflection = final_state['reflection_output']
            logger.info(f"🔍 Final reflection: {'Goal achieved' if reflection.goal_achieved else 'Goal not achieved'}")
            logger.info(f"   Assessment: {reflection.progress_assessment}")
        
        # Success summary
        if final_state.get('current_phase') == AgentPhase.COMPLETE:
            logger.info("🎉 Demo completed successfully!")
        elif final_state.get('current_phase') == AgentPhase.FAILED:
            logger.error("❌ Demo failed")
        else:
            logger.warning("⏸️  Demo stopped early")
            
    except Exception as e:
        logger.error(f"💥 Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
