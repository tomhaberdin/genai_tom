import streamlit as st 
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_experimental.text_splitter import SemanticChunker  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter

# huggingfaceh4/zephyr-7b-alpha (default)
# microsoft/Phi-3-mini-4k-instruct
# pip install pdfplumber 
# pip install faiss-cpu

# REQS
# streamlit
# langchain
# openai
# langchain_community
# langchain_huggingface
# langchain_experimental
# pdfplumber
# faiss-cpu

from streamlit.logger import get_logger
logger = get_logger(__name__)

import os
if os.getenv('USER') == 'appuser':
    hf_token = st.secrets["HF_TOKEN"]
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token
else:
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.environ['MY_HUGGINGFACEHUB_API_TOKEN']

def upload_file(pdf_docs, selected_model):
    docs = []
    for pdf_doc in pdf_docs:
        with open("tmp.pdf", "wb") as f:
           f.write(pdf_doc.getvalue()) 

        logger.info("Loading PDF")
        loader = PDFPlumberLoader("tmp.pdf")
        docs.extend(loader.load())
        logger.info("Done")

    logger.info("Splitting documents")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    documents  = text_splitter.split_documents(docs)
    logger.info("Done")

    emb = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(documents, emb)
    retriever = vector_store.as_retriever()

    # Use the selected model for the LLM
    llm = HuggingFaceEndpoint(repo_id=selected_model)
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain

def main():
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    with st.sidebar:
        st.subheader("Your PDF")
        pdf_docs = st.file_uploader("Please upload your PDF's",
                                    type="pdf",
                                    accept_multiple_files=True)
        
        st.subheader("LLM Model Selection")
        # Let the default model be huggingfaceh4/zephyr-7b-alpha
        selected_model = st.selectbox(
            "Select LLM Model", 
            options=["huggingfaceh4/zephyr-7b-alpha", "microsoft/Phi-3-mini-4k-instruct"],
            index=0
        )
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    qa_chain = upload_file(pdf_docs, selected_model)
                    st.session_state.qa_chain = qa_chain
            else:
                st.error("Please upload at least one PDF file.")

    user_input = st.text_input("Ask your question")

    if user_input:
        if st.session_state.qa_chain is None:
            st.error("Please upload and process a PDF before asking a question.")
        else:
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.invoke(user_input)
                st.write(response)

main()
