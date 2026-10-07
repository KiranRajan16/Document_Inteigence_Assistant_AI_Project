from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from text_splitter import split_documents
from embeddings import get_embedding_model


def get_vector_store():
    # Vector database path
    project_root = Path(__file__).resolve().parent.parent
    vectorstore_path = project_root / "data" / "vectorstore"

    # Load embedding model
    embedding_model = get_embedding_model()

    # Create Chroma vector database
    vector_store = Chroma(
        collection_name="research_documents",
        persist_directory=str(vectorstore_path),
        embedding_function=embedding_model
    )

    return vector_store


def add_uploaded_documents(uploaded_files):
    documents = []
    # Get project root
    project_root = Path(__file__).resolve().parent.parent
    upload_path = project_root / "data" / "documents"

    upload_path.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:

        file_path = upload_path / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(str(file_path))
        pdf_documents = loader.load()

        for document in pdf_documents:
            document.metadata["source_file"] = uploaded_file.name

        documents.extend(pdf_documents)
    # SPlit docs into chunks
    chunks = split_documents(documents)

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    return len(chunks)