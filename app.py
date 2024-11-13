import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With Ollama"
groq_api_key = os.getenv("GROQ_API_KEY")
## Prompt Template
prompt = ChatPromptTemplate(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

def gererate_response(question,api_key,llm,temperature,max_tokens):
    # llm=Ollama(model=llm)
    llm = ChatGroq(model=llm,groq_api_key=groq_api_key)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer
    
## Title of the app
st.title("Enhanced Q&A Chatbot With Ollama")

# st.sidebar.title("Settings")
# api_key=st.sidebar.text_input("Enter your Ollama API Key:",type="password")
    

## Drop down to select various Ollama models
llm=st.sidebar.selectbox("Select an Ollama Large Language Model",["Llama3-8b-8192","Gemma-7b-It","Mixtral-8x7b-32768"])

#Adjust response paramter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    response=gererate_response(user_input,groq_api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")    
