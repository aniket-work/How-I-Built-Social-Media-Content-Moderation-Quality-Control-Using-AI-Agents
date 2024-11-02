from typing import Dict, Any, Optional
from loguru import logger
from json_utils import JSONProcessor, format_grading_response


class GradingProcessor:
    def __init__(self):
        JSONProcessor.setup_logging()
        self.json_processor = JSONProcessor()

    def grade_document(self, client: Any, document: str, question: str) -> Dict[str, str]:
        """Grade document relevance with enhanced error handling."""
        logger.info(f"Grading document for question: {question[:100]}...")

        prompt = f"""Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}.

        Return ONLY a single JSON object with these two keys:
        1. binary_score: Must be either "yes" or "no" 
        2. explanation: A brief explanation

        Important: Return only the JSON object with no additional text or analysis."""

        try:
            result = client.llm.invoke(prompt)
            logger.debug(f"LLM response received: {result.content[:200]}...")

            json_result = self.json_processor.process_llm_response(result.content)
            logger.info("Successfully processed document grading")
            return json_result

        except Exception as e:
            logger.error(f"Error during document grading: {str(e)}")
            return format_grading_response(
                "no",
                f"Error during document grading: {str(e)}"
            )

    def grade_hallucination(
            self,
            client: Any,
            documents: str,
            answer: str
    ) -> Dict[str, str]:
        """Grade for hallucinations with enhanced error handling."""
        logger.info("Starting hallucination grading")
        logger.debug(f"Documents length: {len(documents)}")
        logger.debug(f"Answer length: {len(answer)}")

        prompt = f"""FACTS: \n\n {documents} \n\n STUDENT ANSWER: {answer}

        Return ONLY a single JSON object with these two keys:
        1. binary_score: Must be exactly "yes" or "no" indicating if the answer contains ONLY information from the facts
        2. explanation: A brief explanation of why

        Important: Return only the JSON object. Do not include any additional analysis or multiple JSON objects."""

        try:
            result = client.llm.invoke(prompt)
            logger.debug(f"LLM response for hallucination check: {result.content[:200]}...")

            json_result = self.json_processor.process_llm_response(result.content)
            logger.info("Successfully processed hallucination grading")
            return json_result

        except Exception as e:
            logger.error(f"Error during hallucination check: {str(e)}")
            return format_grading_response(
                "no",
                f"Error during hallucination check: {str(e)}"
            )

    def grade_answer(
            self,
            client: Any,
            question: str,
            answer: str
    ) -> Dict[str, str]:
        """Grade answer quality with enhanced error handling."""
        logger.info(f"Grading answer for question: {question[:100]}...")

        prompt = f"""QUESTION: \n\n {question} \n\n STUDENT ANSWER: {answer}

        Return ONLY a single JSON object with these two keys:
        1. binary_score: Must be exactly "yes" or "no" indicating if the answer addresses the question
        2. explanation: A brief explanation why

        Important: Return only the JSON object. Do not include any additional analysis."""

        try:
            result = client.llm.invoke(prompt)
            logger.debug(f"LLM response for answer grading: {result.content[:200]}...")

            json_result = self.json_processor.process_llm_response(result.content)
            logger.info("Successfully processed answer grading")
            return json_result

        except Exception as e:
            logger.error(f"Error during answer grading: {str(e)}")
            return format_grading_response(
                "no",
                f"Error during answer grading: {str(e)}"
            )