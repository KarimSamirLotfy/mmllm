#!/usr/bin/env python3
"""
Example script demonstrating how to use the LangGraph OCR Agent
"""

import json
import os
from .simple_ocr_agent import SimpleOCRAgent, HistoryItem

def main():
    """Main example function"""
    
    # Set OpenAI API key (make sure you have this in your environment)
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key to use the agent")
        return
    
    # Initialize the agent
    print("Initializing LangGraph OCR Agent...")
    agent = SimpleOCRAgent(model_name="gpt-4o", temperature=0.1)
    
    # Example scenario: Logging into an app
    print("\n=== Example Scenario: App Login ===")
    
    # Step 1: Initial login screen
    print("\nStep 1: Processing login screen...")
    result1 = agent.run_agent(
        image=None,  # In real usage, this would be a PIL Image
        ocr_text="Welcome to MyApp\nUsername: [text field]\nPassword: [text field]\n[Login Button]\nDon't have an account? Sign up",
        ui_description="Login screen with username field at top, password field below it, and a blue login button. Sign up link at bottom.",
        goal="Log into the application with username 'john.doe' and password 'mypassword123'",
        thread_id="login_session_1"
    )
    
    print("Action decided:")
    print(json.dumps(result1.get('action', {}), indent=2))
    
    # Step 2: After typing username, now type password
    print("\nStep 2: Processing after username entry...")
    result2 = agent.run_agent(
        image=None,
        ocr_text="Welcome to MyApp\nUsername: john.doe\nPassword: [text field]\n[Login Button]\nDon't have an account? Sign up",
        ui_description="Login screen with username field filled with 'john.doe', empty password field, and login button available.",
        goal="Log into the application with username 'john.doe' and password 'mypassword123'",
        thread_id="login_session_1"
    )
    
    print("Action decided:")
    print(json.dumps(result2.get('action', {}), indent=2))
    
    # Step 3: After typing password, click login
    print("\nStep 3: Processing after password entry...")
    result3 = agent.run_agent(
        image=None,
        ocr_text="Welcome to MyApp\nUsername: john.doe\nPassword: ********\n[Login Button]\nDon't have an account? Sign up",
        ui_description="Login screen with both username and password fields filled, login button is now highlighted and ready to click.",
        goal="Log into the application with username 'john.doe' and password 'mypassword123'",
        thread_id="login_session_1"
    )
    
    print("Action decided:")
    print(json.dumps(result3.get('action', {}), indent=2))
    
    # Step 4: Success screen
    print("\nStep 4: Processing success screen...")
    result4 = agent.run_agent(
        image=None,
        ocr_text="Welcome john.doe!\nHome\nProfile\nSettings\nLogout",
        ui_description="Main app dashboard showing welcome message and navigation menu with Home, Profile, Settings, and Logout options.",
        goal="Log into the application with username 'john.doe' and password 'mypassword123'",
        thread_id="login_session_1"
    )
    
    print("Action decided:")
    print(json.dumps(result4.get('action', {}), indent=2))
    
    # Get final state to see the history
    print("\n=== Final Agent State ===")
    final_state = agent.get_state("login_session_1")
    
    print(f"Task completed: {final_state.get('task_completed', False)}")
    print(f"Number of history items: {len(final_state.get('history', []))}")
    
    # Print history summary
    print("\nHistory Summary:")
    for i, item in enumerate(final_state.get('history', [])):
        print(f"  Step {i+1}:")
        print(f"    OCR: {item.ocr[:50]}...")
        print(f"    Action: {item.action_taken}")
        print(f"    Task done: {item.task_done}")
        print()


def example_custom_scenario():
    """Example of running a custom scenario"""
    
    print("\n=== Custom Scenario Example ===")
    
    agent = SimpleOCRAgent()
    
    # Custom scenario: Navigate to settings
    result = agent.run_agent(
        image=None,
        ocr_text="Home\nProfile\nSettings\nNotifications\nHelp\nLogout",
        ui_description="Main menu screen with vertical list of options: Home, Profile, Settings, Notifications, Help, and Logout",
        goal="Navigate to the Settings page",
        thread_id="navigation_session"
    )
    
    print("Custom scenario result:")
    print(json.dumps(result.get('action', {}), indent=2))


if __name__ == "__main__":
    main()
    example_custom_scenario()
