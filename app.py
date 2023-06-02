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

HUGGINGFACEHUB_API_TOKEN = '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'

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

# Create instance of OpenAI LLM

repo_id = "google/flan-t5-xl" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})
# llm = OpenAI(temperature=0.1, verbose=True)



st.title('GPT File Interpreter')
# Create a text input box for the user

# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     # To read file as bytes:
#     bytes_data = uploaded_file.getvalue()
#     st.write(bytes_data)

#     # To convert to a string based IO:
#     # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#     # st.write(stringio)


#     loader = PyPDFLoader(uploaded_file)

#     # Can be used wherever a "file-like" object is accepted:
#     # dataframe = pd.read_csv(uploaded_file)
#     # st.write(dataframe)

# Create and load PDF Loader
loader = PyPDFLoader('annualreport.pdf')
# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, collection_name='annualreport')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a banking annual report as a pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    # response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write("response")

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 

