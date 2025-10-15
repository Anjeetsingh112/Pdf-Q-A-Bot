# 🤖 PDF Q&A Bot - AI-Powered Document Intelligence

> Ask questions about your PDF documents and get instant, accurate answers powered by advanced AI models and retrieval-augmented generation (RAG).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.27-green.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Overview

**PDF Q&A Bot** is an intelligent chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about your PDF documents. Upload any PDF, and the bot will index its content, understand your questions, and provide accurate, context-aware answers based solely on the document's information.

### ✨ Key Features

- 📄 **Universal PDF Support** - Upload and analyze any PDF document
- 💬 **Interactive Chat Interface** - Natural conversation-style Q&A with chat history
- 🧠 **Smart Context Retrieval** - Finds the most relevant sections using semantic search
- 🎯 **Accurate Answers** - Powered by Google's FLAN-T5 instruction-tuned model
- ⚡ **Fast & Efficient** - Cached model loading and persistent vector storage
- 🔄 **Multi-Turn Conversations** - Maintains chat history for follow-up questions
- 🎨 **Clean UI** - Intuitive Streamlit interface with real-time responses

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 2GB free disk space (for model downloads)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Anjeetsingh112/Pdf-Q-A-Bot.git
   cd Pdf-Q-A-Bot
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

---

## 📱 How to Use

1. **Upload a PDF**
   - Click the file uploader in the sidebar
   - Select any PDF document from your computer
   - Wait for the indexing process to complete (one-time per PDF)

2. **Ask Questions**
   - Type your question in the chat input at the bottom
   - Press Enter to submit
   - View the AI-generated answer based on your PDF content

3. **Continue the Conversation**
   - Ask follow-up questions
   - Reference previous answers
   - Build contextual conversations

4. **Clear History** (Optional)
   - Click "Clear Chat History" in the sidebar to start fresh

---

## 🧠 How It Works

The PDF Q&A Bot uses a sophisticated RAG (Retrieval-Augmented Generation) pipeline:

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Storage
                                                              ↓
User Question → Question Embedding → Similarity Search → Context Retrieval
                                                              ↓
                                        Context + Question → LLM → Answer
```

### Technical Workflow

1. **Document Processing**
   - PDF is loaded using PyPDFLoader
   - Text is split into 800-character chunks with 150-character overlap
   - Ensures coherent context while fitting model token limits

2. **Vectorization**
   - Text chunks are converted to 384-dimensional vectors
   - Uses `sentence-transformers/all-MiniLM-L6-v2` embeddings
   - Optimized for semantic similarity search

3. **Storage**
   - Vectors stored in ChromaDB (local vector database)
   - Persistent storage for faster reloading
   - Enables quick similarity searches

4. **Retrieval**
   - User question is converted to a vector
   - Top 3 most relevant chunks are retrieved
   - Uses cosine similarity for matching

5. **Generation**
   - Retrieved context + question sent to FLAN-T5
   - Model generates factual, context-based answer
   - Answer cleanup removes artifacts and formatting issues

6. **Display**
   - Clean answer displayed in chat interface
   - Added to conversation history for context

---

## 🔧 Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | Streamlit | Web UI and user interaction |
| **Orchestration** | LangChain | RAG pipeline management |
| **LLM** | FLAN-T5-base | Answer generation |
| **Embeddings** | Sentence Transformers | Text vectorization |
| **Vector DB** | ChromaDB | Semantic search and storage |
| **PDF Parser** | PyPDF | Document text extraction |
| **Model Hub** | HuggingFace | Model hosting and downloads |

---

## 📊 Model Information

### FLAN-T5-base (Google)

- **Parameters**: 220 million
- **Type**: Text-to-text generation (encoder-decoder)
- **Training**: Instruction-tuned on diverse Q&A tasks
- **Size**: ~220MB download (first time only)
- **Inference**: CPU-optimized, fast responses
- **Strengths**: 
  - Instruction following
  - Factual question answering
  - Context comprehension
  - Low hallucination rate

### Embedding Model

- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Speed**: Very fast
- **Quality**: Excellent for semantic search

---

## 💡 Usage Tips & Best Practices

### For Best Results

✅ **Ask specific questions**
```
Good: "What are the three main requirements mentioned in section 2?"
Poor: "Tell me about requirements"
```

✅ **Request specific formats**
```
"Summarize in 3 bullet points"
"Explain in 2 sentences"
"Give a detailed explanation"
```

✅ **Use follow-up questions**
```
1st: "What is the main topic?"
2nd: "Can you elaborate on that?"
3rd: "What are the practical applications?"
```

### Example Questions

- 📌 "What is the main topic of this document?"
- 📌 "Summarize the key findings in 3 sentences"
- 📌 "What methodology was used in this research?"
- 📌 "List all the requirements mentioned"
- 📌 "Explain the concept of [specific term] from the document"
- 📌 "What are the conclusions drawn in the document?"

---

## ⚙️ Configuration

### Environment Variables (Optional)

Create a `.env` file in the root directory:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here  # Optional for model downloads
```

### Customization

Modify `app.py` to customize:

- **Chunk size**: Line 42 (`chunk_size=800`)
- **Chunk overlap**: Line 43 (`chunk_overlap=150`)
- **Retrieved chunks**: Line 75 (`search_kwargs={"k": 3}`)
- **Max answer length**: Line 70 (`max_new_tokens=200`)
- **Model**: Line 68 (`model="google/flan-t5-base"`)

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: Model download fails or is slow
```
Solution: 
- Check internet connection
- First download takes time (~220MB for FLAN-T5)
- Model is cached for future use
```

**Issue**: Slow responses on first question
```
Solution: 
- Normal behavior - model loads into memory
- Subsequent questions are instant (cached)
```

**Issue**: Inaccurate answers
```
Solution:
- Rephrase your question more specifically
- Ensure the information exists in the PDF
- Try: "Based on the document, [your question]"
```

**Issue**: PDF indexing takes long time
```
Solution:
- Large PDFs take longer (one-time process)
- Indexed data is cached (faster on reload)
- Check ChromaDB folder for cached indices
```

**Issue**: Out of memory errors
```
Solution:
- Close other applications
- Try smaller PDFs first
- Reduce chunk size in configuration
```

---

## 📁 Project Structure

```
pdf-qa-bot/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (optional)
├── .gitignore             # Git ignore rules
├── README.md              # This file
│
├── chroma_db_*/           # Vector databases (auto-generated, gitignored)
└── venv/                  # Virtual environment (gitignored)
```

---

## 🔄 Future Enhancements

- [ ] Multi-PDF support (query across multiple documents)
- [ ] PDF page references in answers
- [ ] Export chat history to text/PDF
- [ ] Support for other document formats (DOCX, TXT, HTML)
- [ ] Advanced filtering and search options
- [ ] User authentication and saved sessions
- [ ] API endpoint for programmatic access
- [ ] Larger model options (FLAN-T5-large, etc.)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Anjeet Singh**
- GitHub: [@Anjeetsingh112](https://github.com/Anjeetsingh112)
- Repository: [Pdf-Q-A-Bot](https://github.com/Anjeetsingh112/Pdf-Q-A-Bot)

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) - RAG framework
- [HuggingFace](https://huggingface.co) - Model hosting and transformers
- [Google Research](https://ai.google/research/) - FLAN-T5 model
- [Streamlit](https://streamlit.io) - Web framework
- [ChromaDB](https://www.trychroma.com/) - Vector database

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ using Python, LangChain, and AI**
