import sys
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# project root
project_root = Path(__file__).resolve().parent
# adding src folder
sys.path.append(str(project_root/"src"))

from llm import llm
from graph import graph
from vector_store import add_uploaded_documents
from prompts import summary_prompt, comparison_prompt

# Page configuration
st.set_page_config(
    page_title="AI Research and Document Assistant",
    layout="wide"
)

# title 
st.title("AI Research and Document Assistant")

st.write(
    "Upload and analyze research documents using RAG, LangGraph, LangChain and LLMs"
)

# sidebar 
with st.sidebar:
    st.header("Settings")
    st.write("#### Documents")

    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="upload_files"
    )

    if uploaded_files:

        documents_path = project_root / "data" / "documents"
        existing_files = {
            file.name for file in documents_path.glob("*.pdf")
        }

        new_files = [
            file for file in uploaded_files
            if file.name not in existing_files
        ]

        if not new_files:
            st.warning("All selected documents already exist.")
        else:
            st.write(f"**{len(new_files)} new document(s) selected.**")

            for file in new_files:
                st.write(f"{file.name}")

            if st.button("Add Documents to Knowledge Base"):

                with st.spinner(
                    "Processing documents and updating the vector database..."
                ):
                    try:
                        chunk_count = add_uploaded_documents(new_files)

                        st.success(
                            f"Successfully added {len(new_files)} document(s) "
                            f"with {chunk_count} chunks."
                        )

                    except Exception as e:
                        st.error(f"Error processing documents: {e}")
        
        st.info(
        "The application currently uses the documents stored in the project's data folder"
    )
    st.write("### Available Documents")

    documents_path = project_root/ "data" / "documents"

    if documents_path.exists():
        pdf_files = list(documents_path.glob("*.pdf"))

        for pdf in pdf_files:
            st.write(f"{pdf.name}")
    else:
        st.warning("Documents folder not found")

# Ask Question Section 
st.header("Ask a Question.")

question = st.text_input(
    "Enter your question.",
    placeholder="Example: What is RAG?"
)
if st.button("Ask Question"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching documents and generating answer....."):
            try:
                result = graph.invoke({
                    "question": question,
                    "context": "",
                    "answer": "",
                    "sources": [],
                    "grounded": False,
                    "retry_count":0
                })
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                grounded = result.get("grounded", False)

                st.subheader("Answer")
                if not grounded and answer == "The answer could not be found in the provided documents.":
                    st.warning(answer)
                else:    
                    st.write(answer)

                st.subheader("Grounding Check")
                if grounded:
                    st.success("Answer is supported by the retrieved context.")
                else:
                    st.warning("The answer may not be fully supported by the retrieved context.")

                st.subheader("Sources") 
                if sources:
                    for i, source in enumerate(sources, start=1):
                        st.markdown(
                        f"### Source {i}: {source.get('source', 'Unknown')}"
                        )

                        st.write(
                            f"**Page:** {source.get('page', 'Unknown')}"
                        )

                        with st.expander("View supporting content"):
                            st.write(
                                source.get(
                                    "content",
                                    "Supporting content not available."
                                )
                            )
                else:
                    st.info("No sources were returned")

            except Exception as e:
                st.error(f"Error: {e}")

# Document Summary Section
st.header("Document Summary")

summary_file = st.file_uploader(
    "Select a PDF to summarize",
    type=["pdf"],
    key="summary_file"
)

if summary_file:

    if st.button("Generate Summary"):

        with st.spinner("Reading and summarizing document..."):

            try:
                temp_path = project_root / summary_file.name

                with open(temp_path, "wb") as f:
                    f.write(summary_file.getbuffer())

                loader = PyPDFLoader(str(temp_path))
                documents = loader.load()

                # Combine document pages
                full_text = "\n\n".join(
                    document.page_content
                    for document in documents
                )

                # Split into smaller sections
                max_chars = 12000

                text_chunks = [
                    full_text[i:i + max_chars]
                    for i in range(0, len(full_text), max_chars)
                ]

                summaries = []

                for i, chunk in enumerate(text_chunks):

                    st.write(
                        f"Processing section {i + 1} "
                        f"of {len(text_chunks)}..."
                    )

                    prompt = summary_prompt.format(
                        context=chunk
                    )

                    response = llm.invoke(prompt)

                    summaries.append(response.content)

                # Combine section summaries
                combined_summary = "\n\n".join(summaries)

                st.subheader("Summary")

                # If multiple summaries were generated,
                # create a final concise summary
                if len(summaries) > 1:

                    final_prompt = summary_prompt.format(
                        context=combined_summary
                    )

                    final_response = llm.invoke(final_prompt)

                    st.write(final_response.content)

                else:
                    st.write(combined_summary)

                temp_path.unlink()

            except Exception as e:
                st.error(f"Error generating summary: {e}")

# Document Comparison
st.header("Compare Documents")

comparison_files = st.file_uploader(
    "Select two PDF documents to compare",
    type=["pdf"],
    accept_multiple_files=True,
    key="comparison_files"
)

if comparison_files:

    if len(comparison_files) != 2:
        st.warning("Please select exactly two PDF documents.")

    else:
        if st.button("Compare Documents"):

            with st.spinner("Comparing documents..."):

                try:
                    documents_text = []

                    for file in comparison_files:

                        temp_path = project_root / file.name

                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())

                        loader = PyPDFLoader(str(temp_path))
                        documents = loader.load()

                        text = "\n\n".join(
                            document.page_content
                            for document in documents
                        )

                        documents_text.append(text)

                        temp_path.unlink()

                    document1 = documents_text[0]
                    document2 = documents_text[1]

                    comparison_context = f"""
                            DOCUMENT 1:
                            {document1[:6000]}

                            DOCUMENT 2:
                            {document2[:6000]}
                        """

                    prompt = comparison_prompt.format(
                        context=comparison_context,
                        question="Compare these two documents and explain their similarities, differences, and key concepts."
                    )

                    response = llm.invoke(prompt)

                    st.subheader("Comparison Result")
                    st.write(response.content)

                except Exception as e:
                    st.error(f"Error comparing documents: {e}")