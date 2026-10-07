from pathlib import Path
from langchain_chroma import Chroma
from embeddings import get_embedding_model

def get_vector_store():
    project_root = Path(__file__).resolve().parent.parent
    
    vectorstore_path = project_root/"data"/"vectorstore"

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name = "research_documents",
        persist_directory = str(vectorstore_path),
        embedding_function = embedding_model  
    )

    return vector_store

def get_retriever(k=4):
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs = {"k":k}
    )

    return retriever