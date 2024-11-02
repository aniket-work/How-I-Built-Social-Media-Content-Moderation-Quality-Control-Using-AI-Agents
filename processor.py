from typing import Dict, Any, Optional
from loguru import logger
from graders import GradingProcessor
from json_utils import JSONProcessor


def process_question(
        question: str,
        retriever: Any,
        client: Any,
        context_variables: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Process a question through the RAG pipeline with enhanced error handling and logging.

    Args:
        question: User's question
        retriever: Document retriever instance
        client: LLM client instance
        context_variables: Optional additional context

    Returns:
        Dict containing processing results and any error information
    """
    # Initialize processors
    grading_processor = GradingProcessor()
    logger.info(f"Processing question: {question}")

    try:
        # Retrieve documents
        docs = retriever.invoke(question)
        logger.debug(f"Retrieved {len(docs) if docs else 0} documents")

        # Get document content if available
        doc_txt = docs[1].page_content if docs and len(docs) > 1 else None

        if doc_txt:
            logger.info("Grading retrieved document relevance")
            grade_result = grading_processor.grade_document(client, doc_txt, question)
            logger.debug(f"Document grading result: {grade_result}")

            if grade_result.get("binary_score") == "yes":
                logger.info("Retrieved document is relevant")
                content_source = doc_txt
            else:
                logger.info("Document not relevant, performing web search")
                try:
                    from search import search_web
                    search_results = search_web(question)
                    logger.debug(f"Found {len(search_results)} search results")
                    content_source = "\n".join(search_results)
                except Exception as e:
                    logger.error(f"Web search failed: {str(e)}")
                    content_source = doc_txt  # Fallback to retrieved document
        else:
            logger.info("No documents retrieved, performing web search")
            try:
                from search import search_web
                search_results = search_web(question)
                logger.debug(f"Found {len(search_results)} search results")
                content_source = "\n".join(search_results)
            except Exception as e:
                logger.error(f"Web search failed: {str(e)}")
                return {
                    "error": "No content sources available",
                    "details": str(e)
                }

        # Generate an answer based on the content
        logger.info("Generating answer from content source")
        generation_prompt = f"""Based on this content:
        {content_source[:2000]}  # Limit content length

        Answer this question: {question}

        Provide a clear, concise answer using only information from the content."""

        try:
            answer_response = client.llm.invoke(generation_prompt)
            generated_answer = answer_response.content
            logger.debug(f"Generated answer: {generated_answer[:200]}...")

            # Check for hallucinations
            logger.info("Checking for hallucinations")
            hallucination_result = grading_processor.grade_hallucination(
                client,
                content_source,
                generated_answer
            )
            logger.debug(f"Hallucination check result: {hallucination_result}")

            if hallucination_result.get("binary_score") == "no":
                logger.warning("Hallucination detected in generated answer")
                return {
                    "warning": "Potential hallucination detected",
                    "details": hallucination_result,
                    "original_answer": generated_answer,
                    "content_source": content_source[:500]  # Include excerpt of source
                }

            # Grade the answer
            logger.info("Grading answer quality")
            answer_grade = grading_processor.grade_answer(
                client,
                question,
                generated_answer
            )
            logger.debug(f"Answer grading result: {answer_grade}")

            return {
                "answer": generated_answer,
                "source_type": "retrieved_document" if doc_txt else "web_search",
                "grading_results": {
                    "document_relevance": grade_result if doc_txt else None,
                    "hallucination_check": hallucination_result,
                    "answer_quality": answer_grade
                },
                "content_source": content_source[:500]  # Include excerpt of source
            }

        except Exception as e:
            logger.error(f"Error during answer generation/grading: {str(e)}")
            return {
                "error": "Failed to process answer",
                "details": str(e)
            }

    except Exception as e:
        logger.error(f"Error in process_question: {str(e)}")
        return {
            "error": "Failed to process question",
            "details": str(e)
        }