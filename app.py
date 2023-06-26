# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
# Bring in streamlit for UI/app interface
import streamlit as st
from langchain import PromptTemplate, LLMChain

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
import os

from langchain.agents import create_pandas_dataframe_agent

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from langchain import HuggingFaceHub

import pandas as pd
from io import StringIO


from getpass import getpass

# HUGGINGFACEHUB_API_TOKEN = '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'

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = "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"

# Create instance of OpenAI LLM

# repo_id = "google/flan-t5-xl" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

# llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})
llm = OpenAI(temperature=0.0, verbose=True)


import pandas as pd
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from langchain import PromptTemplate

st.title('AleGent: Your Personal Data Assistant')
# Create a text input box for the user



uploaded_file = st.file_uploader("Choose a file", type={"csv"})
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    agent = create_pandas_dataframe_agent(llm, # Swappable with any LLM model from OpenAI, HuggingFace, GPT4All
                                            df,
                                            verbose=True,
                                            agent_executor_kwargs=agent_kwargs,
                                            return_intermediate_steps=True,
                                            memory = True)
    prompt = PromptTemplate(
    input_variables=["question"],
    template="""
            Instruction: Imagine you are a call-center manager and you have the data for monthly call schedule for call agents that you supervise.
            Context: Use the provided dataframe as context for answering the question.
            Question: {question}
            """,
    )
    str_question = st.text_input('Input your prompt here')
    if str_question:
        # Then pass the prompt to the LLM
        contextual_prompt = prompt.format(question=str_question)
        response = agent({"input": contextual_prompt})
        # ...and write it out to the screen
        st.write(response['output'])

        # With a streamlit expander  
        with st.expander('See thought and action'):
            # Find the relevant pages
            # search = #store.similarity_search_with_score(prompt)
            # Write out the first 
            st.write(response['intermediate_steps']) 

    





# df = pd.read_csv(r"C:\Users\40107907\OneDrive - Anheuser-Busch InBev\AFR Commercial\GenAI\repos\Langchain for docs\LangchainDocuments\call.csv")


# Create and load PDF Loader
# loader = PyPDFLoader('annualreport.pdf')
# Split pages from pdf 



# agent_kwargs = {
#     "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
# }
# memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

# agent = create_pandas_dataframe_agent(llm, # Swappable with any LLM model from OpenAI, HuggingFace, GPT4All
#                                       df,
#                                       verbose=True,
#                                       agent_executor_kwargs=agent_kwargs,
#                                       return_intermediate_steps=True)



# prompt = PromptTemplate(
#     input_variables=["question"],
#     template="""
#             Instruction: Imagine you are a call-center manager and you have the data for monthly call schedule for call agents that you supervise.
#             Context: Use the provided dataframe as context for answering the question.
#             Question: {question}
#             """,
#     )
# str_question = st.text_input('Input your prompt here')


# If the user hits enter
# if str_question:
#     # Then pass the prompt to the LLM
#     contextual_prompt = prompt.format(question=str_question)
#     response = agent({"input": contextual_prompt})
#     # ...and write it out to the screen
#     st.write(response['output'])

#     # With a streamlit expander  
#     with st.expander('See thought and action'):
#         # Find the relevant pages
#         # search = #store.similarity_search_with_score(prompt)
#         # Write out the first 
#         st.write(response['intermediate_steps']) 

