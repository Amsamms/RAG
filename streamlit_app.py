#!/usr/bin/env python3
"""
Streamlit Web Interface for Enhanced PDF RAG System
"""

import streamlit as st
import os
from enhanced_rag_system import EnhancedPDFRAGSystem
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'database_initialized' not in st.session_state:
    st.session_state.database_initialized = False

def initialize_rag_system():
    """Initialize the RAG system"""
    if st.session_state.rag_system is None:
        with st.spinner("🚀 Initializing RAG system..."):
            st.session_state.rag_system = EnhancedPDFRAGSystem()
    return st.session_state.rag_system

def process_pdfs():
    """Process PDF files in the current directory"""
    rag = st.session_state.rag_system
    if rag is None:
        return 0
    
    with st.spinner("📚 Processing PDF files..."):
        chunks_added = rag.process_pdfs(".")
        st.session_state.database_initialized = True
    return chunks_added

def main():
    st.title("📚 PDF RAG System with AI")
    st.markdown("Ask questions about your PDF documents using semantic search and AI!")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Configuration")
        
        # Initialize system
        if st.button("🚀 Initialize System", type="primary"):
            rag = initialize_rag_system()
            st.success("✅ System initialized!")
        
        # Check for PDFs
        pdf_files = list(Path(".").glob("*.pdf"))
        st.subheader("📁 PDF Files Found")
        if pdf_files:
            for pdf in pdf_files:
                st.text(f"📄 {pdf.name}")
        else:
            st.warning("⚠️ No PDF files found in current directory")
        
        # Process PDFs
        if pdf_files and st.button("📚 Process PDFs"):
            if st.session_state.rag_system is None:
                st.error("❌ Please initialize system first")
            else:
                chunks = process_pdfs()
                if chunks > 0:
                    st.success(f"✅ Processed {chunks} text chunks")
                else:
                    st.error("❌ No chunks processed")
        
        # System stats
        if st.session_state.rag_system is not None:
            st.subheader("📊 System Stats")
            stats = st.session_state.rag_system.get_database_stats()
            if 'error' not in stats:
                st.metric("Total Chunks", stats.get('total_chunks', 0))
                st.text(f"🤖 LLM: {'✅ Available' if stats.get('llm_available') else '❌ Not Available'}")
                st.text(f"📐 Embedding: {stats.get('embedding_model', 'Unknown')}")
        
        # API Key configuration
        st.subheader("🔑 API Configuration")
        
        # Show current API key status
        current_key = os.getenv("OPENAI_API_KEY")
        if current_key:
            st.success(f"✅ API Key loaded: {current_key[:10]}...{current_key[-4:]}")
        else:
            st.warning("⚠️ No API key found")
        
        # Allow override
        api_key = st.text_input("Override OpenAI API Key", type="password", help="Enter your OpenAI API key for AI responses")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("✅ API key updated!")
    
    # Main content
    if st.session_state.rag_system is None:
        st.info("👆 Please initialize the system using the sidebar")
        return
    
    # Query interface
    st.header("🔍 Ask Questions")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., 'at what page ethanolamine was mentioned and in what document'",
        help="Ask any question about the content in your PDF documents"
    )
    
    # Search options
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        use_llm = st.checkbox("🤖 Use AI Response", value=True, help="Get natural language answers from AI")
    with col2:
        n_results = st.slider("📊 Results Count", min_value=1, max_value=10, value=5, help="Number of search results to return")
    
    # Search button
    if st.button("🔍 Search", type="primary") and question:
        if not st.session_state.database_initialized:
            st.warning("⚠️ Please process PDF files first using the sidebar")
            return
        
        with st.spinner("🔍 Searching documents..."):
            result = st.session_state.rag_system.ask_question(question, use_llm=use_llm, n_results=n_results)
        
        # Display AI response if available
        if result.get('llm_response'):
            st.subheader("🤖 AI Response")
            st.info(result['llm_response'])
            st.divider()
        
        # Display search results
        st.subheader("📋 Search Results")
        
        if result['search_results']:
            for i, res in enumerate(result['search_results'], 1):
                with st.expander(f"📄 Result {i}: {res['document']} (Page {res['page']})"):
                    st.text(res['text'])
                    st.caption(f"Document: {res['document']} | Page: {res['page']}")
        else:
            st.warning("❌ No relevant documents found for your question")
    
    # Sample questions
    st.header("💡 Sample Questions")
    sample_questions = [
        "at what page ethanolamine was mentioned and in what document",
        "what are the main processes described in the documents",
        "explain the safety procedures mentioned",
        "what chemicals are discussed in the documents",
        "summarize the operating procedures"
    ]
    
    cols = st.columns(2)
    for i, sample in enumerate(sample_questions):
        col = cols[i % 2]
        with col:
            if st.button(f"📝 {sample}", key=f"sample_{i}"):
                st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📚 PDF RAG System - Built with Streamlit, ChromaDB, and OpenAI</p>
        <p>Upload PDF files to the current directory and process them to start asking questions!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()