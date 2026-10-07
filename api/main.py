from fastapi import FastAPI
from langserve import add_routes
from pathlib import Path
from pydantic import BaseModel
import sys

# add src folder path to pyton path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root/"src"))

from graph import graph

# create FastAPI application
app = FastAPI(
    title="AI Research and Document Assistant API",
    description="RAG-based document question answering API",
    version="1.0.0"
)

add_routes(
    app,
    graph,
    path="/rag"
)

# Request Model
class QuestionRequest(BaseModel):
    question: str

# home endpoint
@app.get("/")
def home():
    return{
        "message":"AI Research and document assistant API is running"
    }

# Question endpoint
@app.post("/ask")
def ask_question(request: QuestionRequest):
    result = graph.invoke({
        "question":request.question,
        "context":"",
        "answer":"",
        "sources":[],
        "grounded": False,
        "retry_count": 0
    })

    return{
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "grounded": result["grounded"]
    }