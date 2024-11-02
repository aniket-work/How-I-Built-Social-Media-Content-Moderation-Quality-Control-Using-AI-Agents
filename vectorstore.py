from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings


def create_vectorstore(documents, embedding_model="nomic-embed-text-v1.5"):
    """Create and return a vector store from documents."""
    return SKLearnVectorStore.from_documents(
        documents=documents,
        embedding=NomicEmbeddings(
            model=embedding_model,
            inference_mode="local"
        )
    )