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
import gc

# Load environment variables (optional now, but keep for future)
load_dotenv()

# Page config for Streamlit - MUST BE FIRST
st.set_page_config(
    page_title="PDF Q&A Bot", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Memory optimization: Limit chat history
MAX_CHAT_HISTORY = 10  # Keep only last 10 messages

# Optimized prompt template for FLAN-T5
PROMPT_TEMPLATE = """Answer the question based only on the following context. Be concise and factual.

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# Function to index PDF with memory optimization
@st.cache_resource(show_spinner=False)
def index_pdf(pdf_path, pdf_key):
    """Index PDF and create vector store"""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Reduced to fit FLAN-T5's 512 token limit
            chunk_overlap=150,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        
        # Memory optimization: Use smaller embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=f"./chroma_db_{pdf_key}"
        )
        
        # Clean up
        del documents, splits
        gc.collect()
        
        return vectorstore
    except Exception as e:
        st.error(f"Error indexing PDF: {str(e)}")
        return None

# Function to set up RAG chain with memory optimization
@st.cache_resource(show_spinner=False)
def setup_qa_chain(_vectorstore):
    """Setup QA chain with FLAN-T5 model"""
    try:
        # Use FLAN-T5 with memory optimizations
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=150,  # Reduced from 200
            do_sample=False,  # Greedy decoding for factual answers
            repetition_penalty=1.15,
            device=-1  # Force CPU usage
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_vectorstore.as_retriever(search_kwargs={"k": 2}),  # Reduced from 3
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False  # Don't return source docs to save memory
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error setting up QA chain: {str(e)}")
        return None

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

# Add warning about memory limits
st.sidebar.info("💡 **Tip**: For best performance, use PDFs under 5MB and clear chat history regularly.")

# Sidebar for upload
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf", help="Upload a PDF file (max 10MB)")
pdf_key = None

if uploaded_file is not None:
    # Check file size (limit to 10MB for memory)
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > 10:
        st.error(f"⚠️ File too large ({file_size_mb:.1f}MB). Please upload a PDF smaller than 10MB.")
        st.stop()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    pdf_key = uploaded_file.name  # Unique key for caching
    
    st.sidebar.success(f"✅ PDF uploaded ({file_size_mb:.1f}MB)")
    
    # Index the PDF
    with st.spinner("🔄 Processing PDF... This may take a minute."):
        vectorstore = index_pdf(pdf_path, pdf_key)
    
    if vectorstore is None:
        st.error("Failed to process PDF. Please try again with a different file.")
        st.stop()
    
    st.success("✅ PDF indexed! Now ask questions below.")
    
    # Initialize chat history with limit
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Clear history button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        gc.collect()  # Force garbage collection
        st.rerun()
    
    # Show chat history count
    if len(st.session_state.messages) > 0:
        st.sidebar.text(f"💬 Messages: {len(st.session_state.messages)}/{MAX_CHAT_HISTORY*2}")
    
    # Display chat history
    st.subheader("💬 Chat History")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the PDF:"):
        # Limit chat history to prevent memory overflow
        if len(st.session_state.messages) >= MAX_CHAT_HISTORY * 2:
            st.warning("⚠️ Chat history limit reached. Clearing oldest messages...")
            st.session_state.messages = st.session_state.messages[-(MAX_CHAT_HISTORY * 2 - 2):]
        
        # Append user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Set up QA chain
        qa_chain = setup_qa_chain(vectorstore)
        
        if qa_chain is None:
            st.error("Failed to set up QA chain. Please try again.")
            st.stop()
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Generating answer..."):
                try:
                    result = qa_chain.invoke({"query": prompt})
                    raw_answer = result["result"]
                    # Clean up the answer
                    answer = clean_answer(raw_answer, prompt)
                except Exception as e:
                    answer = f"❌ Error generating answer: {str(e)}"
            
            st.markdown(answer)
        
        # Append assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Clean up
        gc.collect()
        
        # Rerun to update display
        st.rerun()
else:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    
    # Show instructions
    with st.expander("📚 How to Use"):
        st.markdown("""
        1. **Upload a PDF** - Click the file uploader in the sidebar
        2. **Wait for indexing** - The PDF will be processed (one-time)
        3. **Ask questions** - Type your question in the chat input
        4. **Get answers** - AI will respond based on PDF content
        5. **Clear history** - Use the button in sidebar to start fresh
        
        **Tips for best results:**
        - Ask specific questions
        - Use PDFs under 5MB for faster processing
        - Clear chat history if responses slow down
        """)
    
    with st.expander("⚙️ Technical Details"):
        st.markdown("""
        - **Model**: Google FLAN-T5-base (220M parameters)
        - **Embeddings**: Sentence Transformers all-MiniLM-L6-v2
        - **Vector DB**: ChromaDB
        - **Framework**: LangChain + Streamlit
        - **Memory Optimized**: Yes (for cloud deployment)
        """)