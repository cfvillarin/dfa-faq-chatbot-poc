__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

st.title("DFA Chatbot")
st.write(
    "This chatbot answers questions based on FAQs found [here](https://consular.dfa.gov.ph/faqs-menu?). "
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableAssign

from dotenv import load_dotenv
load_dotenv()

from prompts import query_extract_prompt, dfa_rag_prompt

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1)

# RETRIEVER 
CHROMA_PATH = "chroma"
n_retrieved_docs = 5

embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
retriever =  db.as_retriever(search_kwargs={'k': n_retrieved_docs})

def format_docs(docs):
    return f"\n\n".join(f"[FAQ]" + doc.page_content.replace("\n", " ") for n, doc in enumerate(docs, start=1))

chain = query_extract_prompt | llm | {"context": retriever | format_docs, "question": RunnablePassthrough()} | dfa_rag_prompt | llm

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container

    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chain.invoke({"question": prompt})
    # response = f"Echo: {prompt.upper()}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# st.write(message)