# 📚 Multi-Document RAG System with AI Integration

A complete Retrieval-Augmented Generation (RAG) system for querying multiple document types (PDF, DOCX, TXT, XLSX, PPTX) using semantic search and AI-powered responses.

## 🚀 Features

- **📄 Multi-Format Processing**: Supports PDF, DOCX, TXT, XLSX, and PPTX files with automatic text extraction
- **🔍 Semantic Search**: Uses sentence transformers for intelligent document search
- **🤖 AI Responses**: Integrates with OpenAI to provide natural language answers
- **🌐 Web Interface**: Beautiful Streamlit interface for easy interaction
- **📊 Vector Database**: ChromaDB for efficient similarity search
- **📋 Source Citation**: Always shows which document and page contains the information

## 🛠️ Installation

1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key** (optional, for AI responses):
   ```bash
   cp .env.template .env
   # Edit .env and add your OpenAI API key
   ```

## 📁 Setup

1. Place your documents (PDF, DOCX, TXT, XLSX, PPTX) in the project directory
2. The system will automatically detect and process all supported file types

## 🖥️ Usage

### Web Interface (Recommended)

Launch the Streamlit web app:
```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- 🔧 Initialize the RAG system
- 📚 Process multiple document types (PDF, DOCX, TXT, XLSX, PPTX)
- 📁 Upload documents via web interface
- 🔍 Ask questions with AI responses
- 📊 View system statistics
- 💡 Try sample questions

### Command Line Interface

Run the enhanced command-line version:
```bash
python enhanced_rag_system.py
```

### Basic Command Line

Run the original basic version:
```bash
python rag_system.py
```

## 🔑 API Configuration

To enable AI-powered responses:

1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/)
2. Copy `.env.template` to `.env`
3. Add your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## 📝 Example Queries

- "at what page ethanolamine was mentioned and in what document"
- "what are the main safety procedures"
- "explain the chemical processes described"
- "summarize the operating guidelines"
- "what equipment is mentioned in the documents"

## 🏗️ System Architecture

```
📁 Document Files (PDF, DOCX, TXT, XLSX, PPTX)
    ↓
📄 Text Extraction (PyMuPDF, python-docx, openpyxl, python-pptx)
    ↓
✂️ Text Chunking
    ↓
🧮 Embeddings (SentenceTransformers)
    ↓
🗄️ Vector Database (ChromaDB)
    ↓
🔍 Semantic Search
    ↓
🤖 AI Response (OpenAI) + 📋 Citations
```

## 📊 Components

- **PyMuPDF (fitz)**: PDF text extraction with page tracking
- **python-docx**: Word document (.docx) processing
- **openpyxl**: Excel file (.xlsx) processing
- **python-pptx**: PowerPoint (.pptx) processing
- **SentenceTransformers**: Generate embeddings for semantic search
- **ChromaDB**: Vector database for similarity search
- **OpenAI API**: Generate natural language responses
- **Streamlit**: Web interface with file upload capabilities
- **python-dotenv**: Environment variable management

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-3.5-turbo)
- `EMBEDDING_MODEL`: Embedding model (default: all-MiniLM-L6-v2)

### Customization

- **Chunk size**: Modify `chunk_size` in `_split_text()` method
- **Search results**: Adjust `n_results` parameter
- **Embedding model**: Change `EMBEDDING_MODEL` in .env

## 📋 Files

- `streamlit_app.py` - Web interface
- `enhanced_rag_system.py` - Enhanced command-line version with LLM
- `rag_system.py` - Basic RAG system
- `test_ethanolamine.py` - Specific ethanolamine search test
- `extract_ethanolamine_context.py` - Context extraction utility
- `requirements.txt` - Python dependencies
- `.env.template` - Environment variables template

## 🎯 Test Results

**Query**: "at what page ethanolamine was mentioned and in what document"

**Answer**: Ethanolamine was mentioned on pages 37 and 89 in the document "LPG MEROX GOM.unlocked.pdf"

**Context**:
- Page 37: Discussion about amine units using ethanolamine solutions for acid gas extraction
- Page 89: Reference to methyldiethanolamine (MDEA) and other amines in amine water wash processes

## 🚀 Quick Start

1. **Launch web interface**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Initialize system** (click button in sidebar)

3. **Process documents** (click button in sidebar or upload files)

4. **Ask questions** in the main interface

5. **Optional**: Add OpenAI API key for AI responses

## 🆘 Troubleshooting

- **No documents found**: Ensure supported files (PDF, DOCX, TXT, XLSX, PPTX) are in the project directory or upload them via the web interface
- **LLM not working**: Check your OpenAI API key in .env file
- **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Port conflicts**: Streamlit default port is 8501, use `--port` flag to change

## 📄 License

This project is for educational and research purposes.