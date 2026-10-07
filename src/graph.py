from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from retriever import get_retriever
from llm import llm
from prompts import qa_prompt, unanswerable_prompt
from hallucination_check import check_answer_support

# Define state 
class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    sources: list
    grounded: bool
    retry_count: int

# Retrieve Documents
def retrieve_documents(state: GraphState):

    retriever = get_retriever(k=4)

    documents = retriever.invoke(
        state["question"]
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    sources = []

    for document in documents:

        sources.append({
            "source": document.metadata.get(
                "source_file",
                "Unknown"
            ),
            "page": document.metadata.get(
                "page",
                "Unknown"
            ),
            "content": document.page_content
        })

    return {
        "context": context,
        "sources": sources
    }

def check_retrieval(state: GraphState):
    if not state["context"].strip():
        return "not_found"
    return "generate"

# Generate Answer
def generate_answer(state: GraphState):

    prompt = qa_prompt.format(
        context=state["context"],
        question=state["question"]
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }

# Grounding Check
def check_grounding(state: GraphState):

    is_supported = check_answer_support(
        state["context"],
        state["answer"]
    )

    return {
        "grounded": is_supported
    }

# Conditional Routing
def route_after_grounding(state: GraphState):
    if state["grounded"]:
        return "finish"

    if state["retry_count"] >= 1:
        return "not_found"

    return "retry"

# Retry
def retry_generation(state: GraphState):

    return {
        "retry_count": state["retry_count"] + 1
    }

def not_found(state: GraphState):
    return {
        "answer": "The answer could not be found in the provided documents.",
        "grounded": False
    }

# Build Graph
workflow = StateGraph(GraphState)

workflow.add_node(
    "retrieve",
    retrieve_documents
)

workflow.add_node(
    "generate",
    generate_answer
)

workflow.add_node(
    "grounding_check",
    check_grounding
)

workflow.add_node(
    "retry",
    retry_generation
)

workflow.add_node(
    "not_found",
    not_found
)


workflow.add_edge(
    START,
    "retrieve"
)

workflow.add_conditional_edges(
    "retrieve",
    check_retrieval,
    {
        "generate": "generate",
        "not_found": "not_found"
    }
)

workflow.add_edge(
    "generate",
    "grounding_check"
)

workflow.add_edge(
    "not_found",
    END
)


workflow.add_conditional_edges(
    "grounding_check",
    route_after_grounding,
    {
        "finish": END,
        "retry": "retry",
        "not_found": "not_found"
    }
)

workflow.add_edge(
    "retry",
    "generate"
)


graph = workflow.compile()