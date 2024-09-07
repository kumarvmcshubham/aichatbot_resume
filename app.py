from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
from uuid import uuid4

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st


llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    # other params...
)









file_path = "CV.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()






text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)





uuids = [str(uuid4()) for _ in range(len(splits))]

index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
vectorstore = FAISS(model.encode,index,InMemoryDocstore({}),{})
retriever = vectorstore.as_retriever()


vectorstore.add_documents(documents=splits, ids=uuids)





system_prompt = (
    "You are a resume parser who can answer all the questions on the basis of reading a resume.  "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# results = rag_chain.invoke({"input": "What is shubham's main skills?"})

# results










st.title("🦜🔗 Quickstart App")



def generate_response(input_text):
    
    st.info(llm.invoke({"input": input_text}))


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    generate_response(text)
