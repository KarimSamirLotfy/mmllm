
def print_result(result):
    """
    Prints the result of an agent's action in a formatted manner. does not print the image.
    """

    print("\n=== Agent Action Result ===")
    print(f"Action Type: {result.action_type}")
    print(f"Coordinates: {result.coordinates}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Confidence: {result.confidence:.2f}")
    
    if result.action_type == "DUAL_POINT":
        if result.lift_coordinates:
            print("This action is a swipe with start and end points.")
        else:
            print("This action is a tap/click with only a start point.")
    
    print("\n--- End of Result ---\n")