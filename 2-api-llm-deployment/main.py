from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from langchain_community.llms import Ollama
from fastapi import FastAPI
import uvicorn
import os


from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = (
    "true"  # for monitoring on LangSmith paid and locall LLMs
)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="0.1",
    description="API for interacting with LLMs using Langchain",
)

add_routes(app, ChatOpenAI(), path="/openai")

# local LLM
local_llm = Ollama(model="mistral")

prompt1 = ChatPromptTemplate.from_template(
    "Write me am essay about {topic} in 250 words"
)  # interacts with local LLM
prompt2 = ChatPromptTemplate.from_template(
    "Summarize the {topic} in 3 words"
)  # interacts with local LLM

add_routes(app, prompt1 | local_llm, path="/essay")

add_routes(app, prompt2 | local_llm, path="/summary")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
