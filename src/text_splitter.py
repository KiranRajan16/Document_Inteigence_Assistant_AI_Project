from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks