#!/usr/bin/env python3
"""
Demo script for the multi-agent Android in the Wild system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.mmllm.multi_agent import MultiAgentCoordinator
from src.mmllm.utils import EpisodeLoader
from src.mmllm.multi_agent.state import AgentPhase


def run_demo():
    """Run the multi-agent demo."""
    print("🤖 Multi-Agent Android in the Wild Demo")
    print("=" * 50)
    
    # Initialize components
    print("Initializing components...")
    episode_loader = EpisodeLoader()
    coordinator = MultiAgentCoordinator()
    
    # Load test episode
    print("Loading test episode...")
    try:
        initial_state = episode_loader.get_sample_episode_state('google_apps')
        print(f"✅ Loaded real episode: {initial_state['episode_id']}")
    except Exception as e:
        print(f"⚠️  Could not load real episode ({e})")
        print("Using mock episode instead...")
        initial_state = episode_loader._create_mock_state()
    
    print(f"🎯 Goal: {initial_state['goal']}")
    print(f"📱 Max steps: {initial_state['max_steps']}")
    print(f"🎨 UI elements: {len(initial_state['ui_annotations'])}")
    
    # Run multi-agent system
    print("\n🚀 Starting Multi-Agent Execution")
    print("-" * 30)
    
    try:
        final_state = coordinator.run(initial_state)
        
        # Print results
        print("\n📊 Execution Results")
        print("-" * 20)
        print(f"Final phase: {final_state.get('current_phase', 'unknown')}")
        print(f"Steps completed: {final_state.get('current_step', 0)}/{final_state.get('max_steps', 0)}")
        print(f"Errors encountered: {final_state.get('error_count', 0)}")
        
        # Show agent outputs
        if final_state.get('planning_output'):
            planning = final_state['planning_output']
            print(f"🧠 Last planning: {planning.action_type.name} (confidence: {planning.confidence:.2f})")
            print(f"   Reasoning: {planning.reasoning}")
        
        if final_state.get('execution_output'):
            execution = final_state['execution_output']
            print(f"⚡ Last execution: {'Success' if execution.action_executed else 'Failed'}")
            print(f"   Details: {execution.execution_details}")
        
        if final_state.get('reflection_output'):
            reflection = final_state['reflection_output']
            print(f"🔍 Final reflection: {'Goal achieved' if reflection.goal_achieved else 'Goal not achieved'}")
            print(f"   Assessment: {reflection.progress_assessment}")
        
        # Success summary
        if final_state.get('current_phase') == AgentPhase.COMPLETE:
            print("\n🎉 Demo completed successfully!")
        elif final_state.get('current_phase') == AgentPhase.FAILED:
            print("\n❌ Demo failed")
        else:
            print("\n⏸️  Demo stopped early")
            
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
