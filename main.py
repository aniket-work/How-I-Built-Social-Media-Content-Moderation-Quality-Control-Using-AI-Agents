import os
from typing import Dict, Any, Optional
from loguru import logger
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from graders import GradingProcessor
from json_utils import JSONProcessor

# Set environment variables
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    "rag_processing.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="DEBUG",
    rotation="500 MB"
)

# Model configuration
MODEL_NAME = "llama3.2" 
TEMPERATURE = 0


class RAGClient:
    """Wrapper class for RAG processing."""

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)

    def invoke(self, prompt: str) -> str:
        """Invoke the LLM with a prompt."""
        try:
            if isinstance(prompt, str):
                messages = [HumanMessage(content=prompt)]
            else:
                messages = prompt
            response = self.llm(messages)
            return response
        except Exception as e:
            logger.error(f"Error invoking LLM: {str(e)}")
            raise


def setup_vectorstore(urls):
    """Initialize the vector store with documents."""
    try:
        # Load documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Create vectorstore
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        )

        return vectorstore
    except Exception as e:
        logger.error(f"Error setting up vectorstore: {str(e)}")
        raise


def process_question(
        question: str,
        retriever: Any,
        client: RAGClient,
        context_variables: Optional[Dict] = None
) -> Dict[str, Any]:
    """Process a question through the RAG pipeline."""
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

        # Generate answer
        logger.info("Generating answer from content source")
        generation_prompt = f"""Based on this content:
        {content_source[:2000]}

        Answer this question: {question}

        Provide a clear, concise answer using only information from the content."""

        try:
            answer_response = client.invoke(generation_prompt)
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
                    "content_source": content_source[:500]
                }

            # Grade answer
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
                "content_source": content_source[:500]
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


def main():
    """Main execution function."""
    try:
        # Initialize components
        logger.info("Initializing components")
        client = RAGClient()

        urls = [
                "https://www.un.org/en/climatechange/what-is-climate-change",
                "https://www.noaa.gov/education/resource-collections/marine-life"
        ]

        vectorstore = setup_vectorstore(urls)
        retriever = vectorstore.as_retriever(k=3)

        # Test questions
        test_questions = [
            "what is climate change",
            "how species are of Sea turtles?",
        ]

        # Process each question
        for question in test_questions:
            logger.info(f"\nProcessing question: {question}")

            result = process_question(
                question=question,
                retriever=retriever,
                client=client
            )

            # Handle the result
            if "error" in result:
                print(f"Error: {result['error']}")
                print(f"Details: {result['details']}")
            elif "warning" in result:
                print(f"Warning: {result['warning']}")
                print(f"Answer with caveat: {result['original_answer']}")
            else:
                print(f"\nQuestion: {question}")
                print(f"Answer: {result['answer']}")
                print(f"Source: {result['source_type']}")
                print("\nGrading Results:")
                print(f"Document Relevance: {result['grading_results']['document_relevance']}")
                print(f"Hallucination Check: {result['grading_results']['hallucination_check']}")
                print(f"Answer Quality: {result['grading_results']['answer_quality']}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()