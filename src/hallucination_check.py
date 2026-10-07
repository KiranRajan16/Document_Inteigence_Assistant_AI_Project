from llm import llm

def check_answer_support(context, answer):
    prompt = f"""
    You are a factuality checker for a document-based AI assistant.
    Determine whether the answer is fully supported by the provide context. 

    Context:
    {context}

    Answer:
    {answer}

    Rules
    - If the answer is supported by the cotext, respond with SUPPORTED.
    - If the answer is not supported by the context, respond with NOT_SUPPORTED.
    - Do not give any explaination.
    - Respond with only one of these two labels:
        SUPPORTED
        NOT_SUPPORTED
"""
    response = llm.invoke(prompt)

    result = response.content.strip().upper()

    if "NOT_SUPPORTED" in result:
        return False
    if "SUPPORTED" in result:
        return True
    
    return False