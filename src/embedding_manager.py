from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_vector_store(documents, model_name, db_type):

    # Step 1: Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.create_documents(documents)

    # Step 2: Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )

    # Step 3: Create vector database
    if db_type == "FAISS":
        vector_store = FAISS.from_documents(docs, embeddings)

    elif db_type == "Chroma":
        vector_store = Chroma.from_documents(docs, embeddings)

    else:
        raise ValueError("Invalid database selection")

    return vector_store


def semantic_search(vector_store, query, top_k=5):
    """
    Perform semantic search on the vector store
    
    Args:
        vector_store: FAISS or Chroma vector store
        query: User's search query
        top_k: Number of top results to return
        
    Returns:
        List of tuples (document, score)
    """
    # Perform similarity search with scores
    results = vector_store.similarity_search_with_score(query, k=top_k)
    
    return results

