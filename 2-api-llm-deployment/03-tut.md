Langchain Series 03 | Production Grade Deployment LLM As API With Langchain And FastAPI
https://youtu.be/XWB5DXP-DO8

>transcript
yt --transcript https://youtu.be/XWB5DXP-DO8D

>Result
SUMMARY: This video is about creating APIs for LLMs using LangChain. The speaker will discuss how to use LangChain's LangServe library and Fast API to create APIs that can interact with both open source and paid LLM models.

IDEAS:
- Create APIs for large language models (LLMs).
- Use Lang Chain's Lang Serve library.
- Use Fast API.
- Integrate with open source and paid LLM models.
- Create routes for different LLM models.
- Use prompts to specify the task for the LLM model.
- The API can be used by web apps, mobile apps, and other devices.
- This is the first step towards deploying a production-grade LLM application.

INSIGHTS:
- LLMs can be integrated into applications using APIs.
- Lang Chain provides tools for creating LLM APIs.
- APIs can be used to interact with different LLM models.
- Prompts are used to specify the task for the LLM model.
- LLM APIs can be used by various applications.

HABITS:
- Use code comments to explain your code.
- Use a version control system to track changes to your code.
- Test your code thoroughly before deploying it.

FACTS:
- Lang Chain is a library for creating conversational AI applications.
- Fast API is a framework for building web APIs.

JARGON:
- LLM (Large Language Model): A machine learning model trained to generate text, translate languages, write different kinds of creative content, and answer your questions in an informative way.
- API (Application Programming Interface): A way for applications to communicate with each other.
- Prompt: A piece of text that instructs a large language model what to do.

REFERENCES:
- Lang Chain: [https://github.com/jeastham1993/langchain-dotnet](https://github.com/jeastham1993/langchain-dotnet)
- Fast API: [https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker](https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker)

RECOMMENDATIONS:
- Use a cloud server to deploy your LLM API.
- Monitor your LLM API for errors and performance issues.
- Update your LLM API regularly with new features and bug fixes.


## Setup
update requirements.txt
langserve
fastapi
uvicorn

main.py file
import modules and func
  - from fastapi import FastAPI
  - from langchain_core.prompts import ChatPromptTemplate
  - from langchain.chat_models import ChatOpenAI
  - from lanfserve import add_routes
  - import uvicorn
  - import os
  - from langchain_community.llms import Ollama

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true" # for monitoring on LangSmith paid and locall LLMs
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title = "Langchain Server",
    version = "0.1",
    description = "API for interacting with LLMs using Langchain"
)

add_routes(
    app,
    ChatOpenAI(),
    path = "/openai"
)

model = ChatOpenAI()

# local LLM

local_llm = Ollama(model="mistral")

prompt1 = ChatPromptTemplate.from_template("Write me am essay about {topic} in 250 words") # interacts with OpenAI

prompt2 = ChatPromptTemplate.from_template("Summarize the {topic} in 3 sentences") # interacts with local LLM

add_routes(
    app,
    prompt1 | model,
    path = "/essay"
)

add_routes(
    app,
    prompt2 | llm,
    path = "/summary"
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

>cmd
python3 main.py
open http://0.0.0.0:8000/docs in browser
>what use are seeing is swagger docs

>now let's test the API
client.py file
import modules and func
  - import requests
  - import streamlit as st

def get_openai_response(input_text):
    response=requests.post("http://localhost:8000/essay/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']['content']

def get_ollama_response(input_text):
    response=requests.post(
    "http://localhost:8000/summary/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']

# streamlit framework

st.title('Langchain Demo With Ollama API')
input_text=st.text_input("Write an essay on")
input_text1=st.text_input("Summarize the")

if input_text:
    st.write(get_openai_response(input_text))

if input_text1:
    st.write(get_ollama_response(input_text1))


>cmd
streamlit run client.oy