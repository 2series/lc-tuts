Langchain Series 02 | Building Chatbot Using Paid And Open Source LLMs
https://youtu.be/5CJA1Hbutqc

[repo](https://github.com/krishnaik06/Updated-Langchain)

>here we build a `chatbot` as opposed to an `agent`
>chatbot is a `conversational Retrieval Chain`
>this chain allows one to answer follow up questions
>LangSmith to trace and evaluate your language model applications

# Setup
## Template Setup Pipeline venv
conda create -n lc-project python=3.10 -y

conda activate lc-project

pip install -r requirements.txt
langchain_openai 
langchain_core
python-dotenv
streamlit
langchain_community
  
select python interpreter

## project concluded
conda deactivate

conda env remove -n lc-project

.env file
LANGCHAIN_API_KEY="<your-api-key>" # for LangSmith
OPENAI_API_KEY=""
LANGCHAIN_PROJECT="Tutorial1" # for monitoring calls on LangSmith

# Notes
Langchain (LC) is a tool/framework for building LLM applications in Python

LC has numerious modules:
    - paid LLM
    - local LLM

openai-llm.py file
import modules and func
    - from langchain_openai import ChatOpenAI # initialize the model
    - from langchain_core.prompts import ChatPromptTemplate # guides the model response
    - from langchain_core.output_parsers import StrOutputParser # parsing the model response to a string

    - import streamlit as st # web app
    - import os # environment variables
    - from dotenv import load_dotenv # load environment variables

    - os.environ["LANGCHAIN_TRACING_V2"] = "true" # for monitoring on LangSmith paid and locall LLMs
    - os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# First we need a prompt that we can pass into an LLM to generate this search query

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the user's questions"),
        ("user", "{input}")   
    ]
)

# Streamlit tool/framework

st.title("Ollama LLM Demo")
input_text = st.text_input("Enter your text: ")
submit_button = st.button("Submit")

if submit_button:
    response = llm_chain.run(input_text)
    st.write(response)

llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({question: input_text}))



ollama-llm.py file
import modules and func
    - from langchain_community.llms import Ollama # initialize the model
    - from langchain_core.prompts import ChatPromptTemplate # guides the model response
    - from langchain_core.output_parsers import StrOutputParser # parsing the model response to a string

    - import streamlit as st # web app
    - import os # environment variables
    - from dotenv import load_dotenv # load environment variables
    - load_dotenv() # loading all the environment variables

    - os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    - os.environ["LANGCHAIN_TRACING_V2"] = "true" # for monitoring on LangSmith paid and locall LLMs
    - os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# First we need a prompt that we can pass into an LLM to generate this search query

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the user's questions")
        ("user", "{input}")   
    ]
)

# Streamlit tool/framework

st.title("OpenAI LLM Demo")
input_text = st.text_input("Enter your text: ")
submit_button = st.button("Submit")

if submit_button:
    response = llm_chain.run(input_text)
    st.write(response)

llm = Ollama(model="mistral")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({question: input_text}))


[lc-tutorial-repo-2series](https://github.com/2series/lc-tutorial/tree/main)