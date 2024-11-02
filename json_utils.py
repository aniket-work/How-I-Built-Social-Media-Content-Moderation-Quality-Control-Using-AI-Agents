from typing import Dict, Any, Optional, Union
import json
import re
from loguru import logger


class JSONProcessor:
    """Helper class to process and extract JSON from LLM responses with detailed logging."""

    @staticmethod
    def setup_logging():
        """Configure logging with detailed format."""
        logger.remove()  # Remove default handler
        logger.add(
            "json_processing.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="DEBUG"
        )

    @staticmethod
    def clean_response(text: str) -> str:
        """Clean LLM response and remove markdown/code blocks."""
        logger.debug(f"Original text length: {len(text)}")
        logger.debug(f"Original text: {text[:200]}...")  # Log first 200 chars

        # Remove markdown code blocks
        cleaned = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)
        logger.debug(f"After removing code blocks length: {len(cleaned)}")

        # Remove extra whitespace
        cleaned = cleaned.strip()

        logger.debug(f"Final cleaned text length: {len(cleaned)}")
        logger.debug(f"Final cleaned text: {cleaned[:200]}...")

        return cleaned

    @staticmethod
    def extract_last_json(text: str) -> Optional[str]:
        """Extract the last complete JSON object from text."""
        # Find all potential JSON objects
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = list(re.finditer(json_pattern, text))

        logger.debug(f"Found {len(matches)} potential JSON objects")

        if not matches:
            logger.warning("No JSON objects found in text")
            return None

        last_match = matches[-1].group()
        logger.debug(f"Extracted last JSON object: {last_match[:200]}...")
        return last_match

    @staticmethod
    def clean_json_str(json_str: str) -> str:
        """Clean a JSON string for parsing."""
        logger.debug("Cleaning JSON string")

        # Remove any trailing commas before closing braces
        cleaned = re.sub(r',\s*}', '}', json_str)
        # Ensure property names are properly quoted
        cleaned = re.sub(r'(\w+)(?=\s*:)', r'"\1"', cleaned)

        logger.debug(f"Cleaned JSON string: {cleaned}")
        return cleaned

    @staticmethod
    def parse_json(json_str: str) -> Dict[str, Any]:
        """Parse JSON string with multiple fallback attempts."""
        logger.debug("Attempting to parse JSON")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {str(e)}")
            try:
                cleaned = JSONProcessor.clean_json_str(json_str)
                return json.loads(cleaned)
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to parse cleaned JSON: {str(e2)}")
                raise ValueError(f"Failed to parse JSON after cleaning: {json_str}")

    @classmethod
    def process_llm_response(cls, text: str) -> Dict[str, Any]:
        """Process full LLM response to extract and parse JSON."""
        logger.info("Starting LLM response processing")

        # Clean the full response
        cleaned_text = cls.clean_response(text)

        # Extract JSON object
        json_str = cls.extract_last_json(cleaned_text)
        if not json_str:
            logger.error("No JSON object found in cleaned text")
            raise ValueError(f"No JSON object found in text: {text}")

        # Parse the JSON
        try:
            result = cls.parse_json(json_str)
            logger.info("Successfully parsed JSON")
            logger.debug(f"Parsed JSON result: {result}")
            return result
        except ValueError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            raise


def format_grading_response(binary_score: str, explanation: str) -> Dict[str, str]:
    """Create a properly formatted grading response."""
    return {
        "binary_score": binary_score.lower(),
        "explanation": explanation
    }