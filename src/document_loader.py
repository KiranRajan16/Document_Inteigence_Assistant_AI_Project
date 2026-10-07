from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_documents():
    # Get the project root directory  
    project_root = Path(__file__).resolve().parent.parent
    # Path to the pdf docs
    documents_path = project_root / "data" / "documents"

    all_documents = []

    for pdf_file in documents_path.glob("*.pdf"):
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        # add dource filename to metadata 
        for docs in documents:
            docs.metadata["source_file"] = pdf_file.name

        all_documents.extend(documents)
    
    return all_documents

