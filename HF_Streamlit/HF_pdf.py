import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import huggingface_pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
)
st.title("HuggingFace RAG Based LLM App")


st.header("Upload Documents")
pdf_file = st.file_uploader("Upload a PDF File",type=["pdf"])


# Load Documents

def load_documents(pdf_file):
    docs=[]
    if pdf_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            pdf_path = tmp_file.name
        pdf_loader = PyPDFLoader(pdf_path)
        docs.extend(pdf_loader.load())
        os.remove(pdf_path)
    return docs

# split documents
def split_documents(docs,chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(docs)

# Create FAISS Vector Store
def create_vector_store(split_docs):
    embeddings = HuggingFaceEmbeddings(model_id="google/Gemma-Embeddings-v1.0")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store


# Main App
if st.button("Process"):
    if not (pdf_file):
        st.error("Please Upload a PDF File")
    else:
        st.spinner("Processing...")
        documents = load_documents(pdf_file)

        split_docs = split_documents(documents, 1000, 300)
        st.session_state.vector_store = create_vector_store(split_docs)
        st.write("Documents Loaded and Vector Store Created")
        
st.write("Please Enter Your Query about the document")
query = st.text_input("Enter Your Query")
if st.button("Search"):
    if st.session_state.vector_store is None:
        st.error("Please Upload a PDF File")
    elif not query:
        st.error("Please Enter a Query")
    else:
        with st.spinner("Searching..."):
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            llm = huggingface_pipeline(retriever, hf)
            qa_chain = RetrievalQA.from_chain_type(
                llm = llm,
                retriever = retriever,
                chain_type = "stuff",
                return_source_documents = True
            )
            result = qa_chain({"query": query})

            # Display Results
            st.markdown(result["result"])