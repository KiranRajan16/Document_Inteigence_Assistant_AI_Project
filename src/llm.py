import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY is not set in the .env file")

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature = 0,
    groq_api_key = API_KEY
)