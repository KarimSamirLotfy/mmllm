
import logging

logger = logging.getLogger(__name__)

def print_result(result):
    """
    Prints the result of an agent's action in a formatted manner. does not print the image.
    """

    logger.info("=== Agent Action Result ===")
    logger.info(f"Action Type: {result.action_type}")
    logger.info(f"Coordinates: {result.coordinates}")
    logger.info(f"Reasoning: {result.reasoning}")
    logger.info(f"Confidence: {result.confidence:.2f}")
    
    if result.action_type == "DUAL_POINT":
        if result.lift_coordinates:
            logger.debug("This action is a swipe with start and end points.")
        else:
            logger.debug("This action is a tap/click with only a start point.")
    
    logger.info("--- End of Result ---")