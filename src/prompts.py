from langchain_core.prompts import PromptTemplate

# Question Answer Prompt
qa_prompt = PromptTemplate(
    input_variables= ["context","questions"],
    template = """
    You are an AI research and document assistant.
    
    Answer the question using only the information provided in the context below.
    
    Rules:
    - Do not use outside knowledge.
    - Do not make up information.
    - If the answer could not be found in the context, say:
        "I cannot find the answer in the documents provided."
    - Give a clear and consise answer.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
"""
)

# Summarization Prompt
summary_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
    You are an AI research and document assistant.
        
    Summarize the information provided in the context.
        
        Rules:
        - Use only the provided context.
        - Do not use outside information.
        - Preserve the important concepts and key points.
        - Organize the summary clearly. 
        
        Context:
        {context}
        
        Summary:
    """
)

# Document Comparison Prompt
comparison_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a document comparison assistant.

Compare the two documents using ONLY the information provided in the context.

Do not use outside knowledge.
Do not guess or invent information.

Question:
{question}

Context:
{context}

Provide the comparison in a clear and structured format.

Focus on:
- Main topics
- Similarities
- Differences
- Important concepts
- Key takeaways

If the provided context does not contain enough information for a comparison, clearly state that.

Comparison:
"""
)

# Unanswerable Question Prompt
unanswerable_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a document-based AI assistant.

    Answer the question using ONLY the provided context.

    If the answer cannot be found in the context, respond exactly: "The answer could not be found in the provided documents."

    Do not use outside knowledge.
    Do not guess.
    Do not make up information.

    Context:
    {context}

    Question:
    {question}

    Answer:
"""
)