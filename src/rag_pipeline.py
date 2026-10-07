from retriever import get_retriever
from llm import llm
from prompts import qa_prompt, summary_prompt

def ask_question(question):
    # get retriever
    retriever = get_retriever(k=4)
    # retrieve relevant docs
    documents = retriever.invoke(question)

    # combine retrieved content
    context = "\n\n".join(
        document.page_content
        for document in documents 
    )
    # create prompt
    prompt = qa_prompt.format(
        context = context,
        question = question
    )
    # generate answer
    response = llm.invoke(prompt)

    sources = []
    for docs in documents:
        source = docs.metadata.get("source_file","Unknown")
        page = docs.metadata.get("page", "Unknown")

        sources.append({
            "source": source,
            "page": page
        })

    return response.content, sources

def summarize_documents(documents):
    context = "\n\n".join(
        document.page_content
        for document in documents
    )
    prompt = summary_prompt.format(
        context = context
    )
    response = llm.invoke(prompt)

    return response.content