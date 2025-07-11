#!/usr/bin/env python3
"""
Example script demonstrating the simplified one-node LangGraph OCR Agent
Flow: start -> model -> wait for next episode -> model -> ... -> end
"""

import json
import os
from mmllm.agent_ocr.simple_ocr_agent import SimpleOCRAgent, HistoryItem

def main():
    """Main example function demonstrating step-by-step execution"""
        
    # Initialize the agent
    print("Initializing Simple LangGraph OCR Agent...")
    agent = SimpleOCRAgent()
    
    # Example scenario: Step-by-step login process
    print("\n=== Step-by-Step Login Scenario ===")
    
    thread_id = "login_session_1"
    goal = "Log into the application with username 'john.doe' and password 'mypassword123'"
    
    # Episode 1: Initial login screen
    print("\n--- Episode 1: Initial Login Screen ---")
    result1 = agent.run_step(
        image=None,  # In real usage, this would be a PIL Image
        ocr_text="Welcome to MyApp\nUsername: [text field]\nPassword: [text field]\n[Login Button]\nDon't have an account? Sign up",
        ui_description="Login screen with empty username field at top, empty password field below it, and a blue login button. Sign up link at bottom.",
        goal=goal,
        thread_id=thread_id
    )
    
    print("Agent Action:")
    print(json.dumps(result1.get('action', {}), indent=2))
    print(f"Task completed: {result1.get('task_completed', False)}")
    
    # Check if task is done
    if result1.get('task_completed', False):
        print("Task completed!")
        return
    
    print("\n👤 User executes the action and provides next episode...")
    
    # Episode 2: After typing username
    print("\n--- Episode 2: After Username Entry ---")
    result2 = agent.run_step(
        image=None,
        ocr_text="Welcome to MyApp\nUsername: john.doe\nPassword: [text field]\n[Login Button]\nDon't have an account? Sign up",
        ui_description="Login screen with username field filled with 'john.doe', empty password field highlighted, and login button available.",
        goal=goal,
        thread_id=thread_id
    )
    
    print("Agent Action:")
    print(json.dumps(result2.get('action', {}), indent=2))
    print(f"Task completed: {result2.get('task_completed', False)}")
    
    # Check if task is done
    if result2.get('task_completed', False):
        print("Task completed!")
        return
    
    print("\n👤 User executes the action and provides next episode...")
    
    # Episode 3: After typing password
    print("\n--- Episode 3: After Password Entry ---")
    result3 = agent.run_step(
        image=None,
        ocr_text="Welcome to MyApp\nUsername: john.doe\nPassword: ********\n[Login Button]\nDon't have an account? Sign up",
        ui_description="Login screen with both username and password fields filled, login button is now highlighted and ready to click.",
        goal=goal,
        thread_id=thread_id
    )
    
    print("Agent Action:")
    print(json.dumps(result3.get('action', {}), indent=2))
    print(f"Task completed: {result3.get('task_completed', False)}")
    
    # Check if task is done
    if result3.get('task_completed', False):
        print("Task completed!")
        return
    
    print("\n👤 User executes the action and provides next episode...")
    
    # Episode 4: Success screen after login
    print("\n--- Episode 4: Success Screen ---")
    result4 = agent.run_step(
        image=None,
        ocr_text="Welcome john.doe!\nDashboard\nProfile\nSettings\nLogout",
        ui_description="Main app dashboard showing welcome message with user's name and navigation menu with Dashboard, Profile, Settings, and Logout options.",
        goal=goal,
        thread_id=thread_id
    )
    
    print("Agent Action:")
    print(json.dumps(result4.get('action', {}), indent=2))
    print(f"Task completed: {result4.get('task_completed', False)}")
    
    # Get final state to see the complete history
    print("\n=== Final Agent State ===")
    final_state = agent.get_state(thread_id)
    
    print(f"Task completed: {final_state.get('task_completed', False)}")
    print(f"Number of history items: {len(final_state.get('history', []))}")
    
    # Print complete history
    print("\nComplete History:")
    for i, item in enumerate(final_state.get('history', [])):
        print(f"  Episode {i+1}:")
        print(f"    OCR: {item.ocr[:50]}...")
        print(f"    UI: {item.ui_description[:50]}...")
        print(f"    Action Type: {item.action_taken.get('action_type', 'N/A')}")
        print(f"    Coordinates: {item.action_taken.get('coordinates', 'N/A')}")
        print(f"    Text: {item.action_taken.get('text', 'N/A')}")
        print(f"    Step completed: {item.task_done}")
        print()


if __name__ == "__main__":
    main()