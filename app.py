import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from transformers import pipeline
import tempfile

# Load environment variables (optional now, but keep for future)
load_dotenv()

# Page config for Streamlit
st.set_page_config(page_title="PDF Q&A Bot", page_icon="🤖", layout="wide")

# Optimized prompt template for FLAN-T5
PROMPT_TEMPLATE = """Answer the question based only on the following context. Be concise and factual.

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# Function to index PDF (unchanged, but add key for session)
@st.cache_resource
def index_pdf(pdf_path, pdf_key):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Reduced to fit FLAN-T5's 512 token limit
        chunk_overlap=150,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=f"./chroma_db_{pdf_key}"  # Unique per PDF for multi-file future
    )
    # Note: Chroma 0.4.x+ auto-persists, no need to call persist()
    
    return vectorstore

# Function to set up RAG chain with FLAN-T5 (much better than GPT-2)
@st.cache_resource  # Cache to avoid reloading model on every question
def setup_qa_chain(_vectorstore):
    # Use FLAN-T5: instruction-tuned model specifically designed for Q&A
    pipe = pipeline(
        "text2text-generation",  # FLAN-T5 uses text2text
        model="google/flan-t5-base",  # 220M params, fast and accurate
        max_new_tokens=200,
        do_sample=False,  # Greedy decoding for factual answers
        repetition_penalty=1.15,
        # Note: temperature is ignored when do_sample=False (greedy decoding)
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# Function to clean up model output
def clean_answer(answer, question):
    """Remove prompt artifacts and extract clean answer"""
    answer = answer.strip()
    
    # Remove if answer starts with the question
    if answer.lower().startswith(question.lower()):
        answer = answer[len(question):].strip()
    
    # Remove "Answer:" prefix if present
    if answer.lower().startswith("answer:"):
        answer = answer[7:].strip()
    
    # Remove prompt artifacts
    artifacts = ["Context:", "Question:", "You are a helpful"]
    for artifact in artifacts:
        if artifact in answer:
            answer = answer.split(artifact)[0].strip()
    
    # Take only first paragraph
    paragraphs = answer.split('\n\n')
    if paragraphs:
        answer = paragraphs[0].strip()
    
    return answer if answer else "I don't have enough information to answer this question."


# Streamlit UI
st.title("🤖 AI-Powered PDF Q&A Bot")
st.markdown("Upload a PDF to start asking questions! Powered by LangChain, HuggingFace Transformers, and ChromaDB.")

# Sidebar for upload
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
pdf_key = None

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    pdf_key = uploaded_file.name  # Unique key for caching
    
    st.sidebar.success("PDF uploaded! Indexing...")
    
    # Index the PDF
    with st.spinner("Processing PDF... This may take a minute."):
        vectorstore = index_pdf(pdf_path, pdf_key)
    
    st.success("PDF indexed! Now ask questions below.")
    
    # New: Chat History Setup (multi-turn)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Clear history on new PDF (optional: add button)
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
    
    # Display chat history
    st.subheader("Chat History")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # New: Chat-style question input (after history)
    if prompt := st.chat_input("Ask a question about the PDF:"):
        # Append user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Set up QA chain
        qa_chain = setup_qa_chain(vectorstore)
        
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                result = qa_chain.invoke({"query": prompt})
                raw_answer = result["result"]
                # Clean up the answer
                answer = clean_answer(raw_answer, prompt)
            
            st.markdown(answer)
        
        # Append assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Rerun to update display
        st.rerun()
else:
    st.info("👈 Upload a PDF in the sidebar to get started.")