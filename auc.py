import os
import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import requests
from langchain.llms import HuggingFaceHub
from langchain.embeddings import GPT4AllEmbeddings
from PIL import Image
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
#import fitz

# Adding OpenAI's API Key as environment variable
os.environ["OPENAI_API_KEY"] = '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'
# 1. Indexing: Load, Split, Embed and Store

# instantiating a TextLoader instance, giving it the file name as input as well as the encoding
loader = TextLoader("combined_text.txt", encoding='utf-8')

# loading all the documents and their metadata, since we only have a single document, it loads it
# and adds some metadata such as linenumbers 
docs = loader.load()

# instantiating a RecursiveTextSplitter object 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# splitting the document instances into chunks of at most 1000 characters with a chunk overlap of 200 characters
# the chunk overalpping is used to ensure the coherence of semantic understanding, so that the model understands that 
# the last 200 characters are related to the next 1000 characters, to enhance context understanding 
# and stores them into a variable called splits
splits = text_splitter.split_documents(docs)

# utilizes chromadb as our vector database or vector store, and using OpenAI's embedding model
vectorstore = Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings())
# 2. Retrieval and Generation

# instantiates a VectorStoreRetriever object from Langchain
# this as_retriever() method is responsible for similarity search
# we are currently utilizing the default search type called "similarity" which is based on cosine similarity
retriever = vectorstore.as_retriever()

# our customized template for the prompt
template = """
You are a friendly assistant for question-answering tasks. Greet people when they tell you your name. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
            """
# making the template our prompt 
prompt = ChatPromptTemplate.from_template(template)

# using gpt-3.5-turbo as our underlying model with temperature set to 0.2
# temperature controls creativity, it's a value from 0 - 1
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
#llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", huggingfacehub_api_token='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', model_kwargs={"temperature": 0.1, "max_length": 4096})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# chain expression, retriever fetches the relevant documents from indexed chunks, 
# and the RunnablePassThrough() just sends the users query as it is to the model
# the prompt and llm choices are passed to the chain
# and finally the StrOutParser() parses the response of the llm as a string 
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
def main():
    st.title("AUC Policy & Procedures RAG AI Bot")

    # Read the image file
    image = Image.open('0.png')
        
    # Display the image using st.image
    st.image(image)
    user_input = st.text_input("Enter your input:")
    # Check if user has entered any input
    if user_input:
        # Process the input and get the model output
        response = rag_chain.invoke(user_input)   
        # Display the model output
        st.write("Model Output:")
        st.write(response)
        #print(chat_history)
# Run the Streamlit app
if __name__ == "__main__":
    main()