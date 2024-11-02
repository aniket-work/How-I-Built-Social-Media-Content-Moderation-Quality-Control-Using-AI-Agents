import unittest
from loguru import logger
from typing import Dict, Any
import json


class TestJSONProcessor:
    """Helper class to process and extract JSON from LLM responses with detailed logging."""

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """Extract the last valid JSON object from text."""
        logger.debug(f"Processing text: {text[:200]}...")

        # Split on code blocks if present
        parts = text.split("```")

        # Find JSON objects in each part
        json_objects = []
        for part in parts:
            try:
                # Find start and end of JSON object
                start = part.find("{")
                end = part.rfind("}") + 1
                if start >= 0 and end > 0:
                    json_str = part[start:end]
                    # Try to parse JSON
                    json_obj = json.loads(json_str)
                    json_objects.append(json_obj)
                    logger.debug(f"Found valid JSON: {json_str}")
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse JSON in part: {part[:100]}...")
                continue

        if not json_objects:
            raise ValueError(f"No valid JSON objects found in: {text}")

        # Return the last valid JSON object found
        logger.info("Successfully extracted last JSON object")
        return json_objects[-1]


class TestJSONProcessing(unittest.TestCase):
    """Test cases for JSON processing logic."""

    def setUp(self):
        self.processor = TestJSONProcessor()
        self.maxDiff = None  # Show full diffs in test output

    def test_single_json_response(self):
        """Test processing a single JSON response."""
        input_text = '''
        {
            "binary_score": "yes",
            "explanation": "Simple explanation"
        }
        '''
        expected = {
            "binary_score": "yes",
            "explanation": "Simple explanation"
        }
        result = self.processor.extract_json(input_text)
        self.assertEqual(result, expected)

    def test_multiple_json_response(self):
        """Test processing response with multiple JSON objects."""
        input_text = '''
        Here's the JSON response:
        ```
        {
            "binary_score": "yes",
            "explanation": "First explanation"
        }
        ```

        However, upon closer inspection:

        ```
        {
            "binary_score": "no",
            "explanation": "Second explanation"
        }
        ```
        '''
        expected = {
            "binary_score": "no",
            "explanation": "Second explanation"
        }
        result = self.processor.extract_json(input_text)
        self.assertEqual(result, expected)

    def test_real_llm_response(self):
        """Test processing actual LLM response pattern."""
        input_text = '''Here's the JSON response:
        ```
        {
          "binary_score": "yes",
          "explanation": "The student answer mentions specific LLM models (Llama 3.2 11B Vision Instruct and Llama 3.2 90B Vision Instruct) that are available on Azure AI Model Catalog, which is a direct reference to the FACTS about external tools for multimodal AI."
        }
        ```

        However, upon closer inspection, I noticed that the student answer also mentions specific models (Claude 3 Haiku and GPT-4o mini) from other companies (Anthropic and OpenAI), but these are not mentioned in the provided FACTS. Therefore, I would revise the binary_score to "no" and provide a more detailed explanation:

        ```
        {
          "binary_score": "no",
          "explanation": "The student answer mentions specific LLM models from other companies (Claude 3 Haiku and GPT-4o mini) that are not mentioned in the provided FACTS. While it does mention Meta's Llama 3.2 models, which is grounded in the FACTS, the additional information about other models is not supported by the provided text."
        }
        ```'''

        expected = {
            "binary_score": "no",
            "explanation": "The student answer mentions specific LLM models from other companies (Claude 3 Haiku and GPT-4o mini) that are not mentioned in the provided FACTS. While it does mention Meta's Llama 3.2 models, which is grounded in the FACTS, the additional information about other models is not supported by the provided text."
        }
        result = self.processor.extract_json(input_text)
        self.assertEqual(result, expected)

    def test_malformed_json(self):
        """Test handling of malformed JSON."""
        input_text = '''
        {
            "binary_score": "yes"
            "explanation": "Missing comma"
        }
        '''
        with self.assertRaises(ValueError):
            self.processor.extract_json(input_text)


if __name__ == '__main__':
    # Configure logging
    logger.remove()
    logger.add("test_json_processing.log", level="DEBUG")

    # Run tests
    unittest.main(argv=[''], verbosity=2)