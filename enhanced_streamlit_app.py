#!/usr/bin/env python3
"""
Enhanced Streamlit App with Document Selection and Multi-Format Support
Users can choose specific documents by category and control processing
"""

import streamlit as st
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any
import time
from datetime import datetime
from dotenv import load_dotenv

# Import our scalable system
from scalable_rag_system import ScalableMultiFormatRAG
from file_upload_manager import FileUploadManager, display_file_upload_interface

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Enhanced PDF RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {}

def get_file_info(directory: str = ".", include_uploaded: bool = True) -> Dict[str, List[Dict]]:
    """Get information about all supported files in directory"""
    file_categories = {
        'PDF': {'extensions': ['.pdf'], 'files': []},
        'Word': {'extensions': ['.docx', '.doc'], 'files': []},
        'Excel': {'extensions': ['.xlsx', '.xls'], 'files': []},
        'PowerPoint': {'extensions': ['.pptx', '.ppt'], 'files': []},
    }
    
    # Scan directory for files
    for category, info in file_categories.items():
        for ext in info['extensions']:
            files = list(Path(directory).glob(f"*{ext}"))
            for file in files:
                file_stat = file.stat()
                file_info = {
                    'name': file.name,
                    'path': str(file),
                    'size': file_stat.st_size,
                    'modified': datetime.fromtimestamp(file_stat.st_mtime),
                    'category': category,
                    'extension': ext,
                    'source': 'local'
                }
                info['files'].append(file_info)
    
    # Include uploaded files if requested
    if include_uploaded:
        try:
            upload_manager = FileUploadManager()
            uploaded_files = upload_manager.list_uploaded_files()
            
            for category, uploaded_list in uploaded_files.items():
                if category in file_categories:
                    for file_info in uploaded_list:
                        file_info['source'] = 'uploaded'
                        file_info['modified'] = datetime.fromtimestamp(Path(file_info['path']).stat().st_mtime)
                        file_categories[category]['files'].append(file_info)
        except Exception as e:
            st.warning(f"Could not load uploaded files: {e}")
    
    return file_categories

def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

def display_file_selection():
    """Display file selection interface"""
    st.header("📁 Document Selection & Processing")
    
    # Get file information
    file_categories = get_file_info(".")
    
    # Display file categories and selection
    selected_files = []
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Available Documents by Category")
        
        for category, info in file_categories.items():
            if info['files']:
                with st.expander(f"📊 {category} Files ({len(info['files'])} found)", expanded=True):
                    
                    # Category controls
                    col_select, col_limit = st.columns([1, 1])
                    
                    with col_select:
                        select_all = st.checkbox(f"Select All {category}", key=f"select_all_{category}")
                    
                    with col_limit:
                        max_files = st.number_input(
                            f"Max {category} files to process", 
                            min_value=0, 
                            max_value=len(info['files']), 
                            value=len(info['files']),
                            key=f"max_{category}"
                        )
                    
                    # File list with individual selection
                    for i, file_info in enumerate(info['files'][:max_files]):
                        col_check, col_info, col_size, col_date, col_source = st.columns([1, 2.5, 1, 1.5, 1])
                        
                        with col_check:
                            is_selected = st.checkbox(
                                "", 
                                value=select_all,
                                key=f"{category}_{i}_{file_info['name']}"
                            )
                            if is_selected:
                                selected_files.append(file_info)
                        
                        with col_info:
                            st.text(file_info['name'])
                        
                        with col_size:
                            st.text(format_file_size(file_info['size']))
                        
                        with col_date:
                            st.text(file_info['modified'].strftime("%Y-%m-%d %H:%M"))
                        
                        with col_source:
                            source_emoji = "📁" if file_info.get('source') == 'local' else "📤"
                            st.text(f"{source_emoji} {file_info.get('source', 'local').title()}")
            else:
                st.info(f"No {category} files found in current directory")
    
    with col2:
        st.subheader("📊 Selection Summary")
        
        # Summary statistics
        if selected_files:
            summary_data = {}
            total_size = 0
            
            for file_info in selected_files:
                category = file_info['category']
                if category not in summary_data:
                    summary_data[category] = {'count': 0, 'size': 0}
                summary_data[category]['count'] += 1
                summary_data[category]['size'] += file_info['size']
                total_size += file_info['size']
            
            st.markdown("**Selected Files:**")
            for category, data in summary_data.items():
                st.metric(
                    f"{category} Files",
                    data['count'],
                    f"{format_file_size(data['size'])}"
                )
            
            st.markdown("---")
            st.metric("Total Files", len(selected_files))
            st.metric("Total Size", format_file_size(total_size))
            
            # Processing options
            st.subheader("⚙️ Processing Options")
            
            chunk_size = st.slider("Chunk Size (characters)", 200, 1000, 500)
            max_workers = st.slider("Parallel Workers", 1, 8, 4)
            batch_size = st.slider("Batch Size", 10, 100, 50)
            
            # Process button
            if st.button("🚀 Process Selected Documents", type="primary"):
                process_selected_files(selected_files, chunk_size, max_workers, batch_size)
        else:
            st.info("Select documents to see processing options")
    
    return selected_files

def process_selected_files(selected_files: List[Dict], chunk_size: int, max_workers: int, batch_size: int):
    """Process the selected files"""
    if not selected_files:
        st.error("No files selected for processing")
        return
    
    # Initialize RAG system if needed
    if st.session_state.rag_system is None:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = ScalableMultiFormatRAG(batch_size=batch_size)
    
    # Processing progress
    progress_container = st.container()
    
    with progress_container:
        st.subheader("🔄 Processing Documents")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each file
        total_chunks = 0
        processing_stats = {
            'total_files': len(selected_files),
            'successful': 0,
            'failed': 0,
            'total_chunks': 0,
            'processing_time': 0,
            'categories': {}
        }
        
        start_time = time.time()
        
        for i, file_info in enumerate(selected_files):
            # Update progress
            progress = (i + 1) / len(selected_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {file_info['name']} ({i+1}/{len(selected_files)})")
            
            try:
                # Process single file
                file_chunks = st.session_state.rag_system.process_single_file(Path(file_info['path']))
                
                if file_chunks:
                    # Add to database
                    rag_system = st.session_state.rag_system
                    
                    documents = []
                    embeddings = []
                    metadatas = []
                    ids = []
                    
                    for chunk_data in file_chunks:
                        # Create embeddings
                        embedding = rag_system.embedding_model.encode(chunk_data['text'])
                        
                        # Prepare data
                        doc_id = f"{chunk_data['document']}_page_{chunk_data['page']}_chunk_{chunk_data['chunk_id']}"
                        
                        documents.append(chunk_data['text'])
                        embeddings.append(embedding.tolist())
                        metadatas.append({
                            'document': chunk_data['document'],
                            'page': chunk_data['page'],
                            'chunk_id': chunk_data['chunk_id'],
                            'file_path': chunk_data['file_path'],
                            'file_type': chunk_data['file_type']
                        })
                        ids.append(doc_id)
                    
                    # Add to database
                    rag_system.collection.add(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                    
                    total_chunks += len(file_chunks)
                    processing_stats['successful'] += 1
                    
                    # Update category stats
                    category = file_info['category']
                    if category not in processing_stats['categories']:
                        processing_stats['categories'][category] = {'files': 0, 'chunks': 0}
                    processing_stats['categories'][category]['files'] += 1
                    processing_stats['categories'][category]['chunks'] += len(file_chunks)
                    
                else:
                    processing_stats['failed'] += 1
                    st.warning(f"No content extracted from {file_info['name']}")
                    
            except Exception as e:
                processing_stats['failed'] += 1
                st.error(f"Error processing {file_info['name']}: {str(e)}")
        
        # Complete processing
        processing_stats['total_chunks'] = total_chunks
        processing_stats['processing_time'] = time.time() - start_time
        
        # Update session state
        st.session_state.processed_files.extend(selected_files)
        st.session_state.processing_stats = processing_stats
        
        # Show results
        progress_bar.progress(1.0)
        status_text.text("✅ Processing Complete!")
        
        # Display results
        st.success(f"🎉 Successfully processed {processing_stats['successful']} files!")
        st.info(f"📊 Total chunks created: {processing_stats['total_chunks']}")
        st.info(f"⏱️ Processing time: {processing_stats['processing_time']:.2f} seconds")
        
        # Category breakdown
        if processing_stats['categories']:
            st.subheader("📋 Processing Summary by Category")
            for category, stats in processing_stats['categories'].items():
                st.metric(
                    f"{category}",
                    f"{stats['files']} files",
                    f"{stats['chunks']} chunks"
                )

def display_query_interface():
    """Display the query interface"""
    if st.session_state.rag_system is None:
        st.info("Please process some documents first using the Document Selection tab")
        return
    
    st.header("🔍 Query Your Documents")
    
    # Get database stats
    try:
        stats = st.session_state.rag_system.get_database_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Chunks", stats.get('total_chunks', 0))
        with col2:
            st.metric("LLM Available", "✅ Yes" if stats.get('llm_available') else "❌ No")
        with col3:
            st.metric("Processed Files", len(st.session_state.processed_files))
    except:
        st.warning("Unable to get database statistics")
    
    # Query input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., 'where is ethanolamine mentioned?'",
        help="Ask any question about your processed documents"
    )
    
    # Query options
    col1, col2, col3 = st.columns(3)
    with col1:
        use_llm = st.checkbox("🤖 Use AI Response", value=True)
    with col2:
        n_results = st.slider("Number of Results", 1, 20, 5)
    with col3:
        file_filter = st.selectbox(
            "Filter by File Type",
            ["All Files", "PDF", "Word", "Excel", "PowerPoint"]
        )
    
    # Search button
    if st.button("🔍 Search", type="primary") and question:
        with st.spinner("Searching documents..."):
            try:
                # Perform search
                results = st.session_state.rag_system.search_documents(question, n_results)
                
                # Filter by file type if specified
                if file_filter != "All Files":
                    filtered_results = []
                    for result in results['results']:
                        if result.get('file_type', '').lower() == file_filter.lower():
                            filtered_results.append(result)
                    results['results'] = filtered_results
                
                # Display results
                if results['results']:
                    # AI Response
                    if use_llm:
                        ai_response = st.session_state.rag_system.generate_llm_response(
                            question, results['results'][:5]
                        )
                        if ai_response:
                            st.subheader("🤖 AI Response")
                            st.info(ai_response)
                            st.divider()
                    
                    # Search Results
                    st.subheader(f"📋 Search Results ({len(results['results'])} found)")
                    
                    for i, result in enumerate(results['results'], 1):
                        with st.expander(f"📄 Result {i}: {result['document']} (Page {result['page']}) - {result.get('file_type', 'unknown').title()}"):
                            st.text(result['text'])
                            
                            # Metadata
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.caption(f"📄 Document: {result['document']}")
                            with col2:
                                st.caption(f"📄 Page: {result['page']}")
                            with col3:
                                st.caption(f"📊 Similarity: {1 - result.get('distance', 0):.3f}")
                else:
                    st.warning("No relevant documents found for your query")
                    
            except Exception as e:
                st.error(f"Search error: {str(e)}")

def display_cost_estimator():
    """Display cost estimation interface"""
    st.header("💰 Cost Estimator")
    
    st.subheader("📊 Usage Estimation")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        queries_per_month = st.number_input("Queries per month", 1, 100000, 100)
        model_selection = st.selectbox(
            "OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"]
        )
    
    with col2:
        context_size = st.slider("Average context size (tokens)", 500, 4000, 2000)
        response_length = st.slider("Average response length (tokens)", 50, 500, 200)
    
    # Calculate costs
    pricing = {
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-4': {'input': 0.03, 'output': 0.06}
    }
    
    model_prices = pricing[model_selection]
    
    # Cost calculation
    total_input_tokens = (context_size + 20) * queries_per_month  # +20 for question
    total_output_tokens = response_length * queries_per_month
    
    input_cost = (total_input_tokens / 1000) * model_prices['input']
    output_cost = (total_output_tokens / 1000) * model_prices['output']
    total_cost = input_cost + output_cost
    
    # Display results
    st.subheader("💰 Cost Breakdown")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monthly Cost", f"${total_cost:.2f}")
    with col2:
        st.metric("Cost per Query", f"${total_cost/queries_per_month:.4f}")
    with col3:
        st.metric("Annual Cost", f"${total_cost*12:.2f}")
    
    # Comparison table
    st.subheader("📈 Model Comparison")
    
    comparison_data = []
    for model, prices in pricing.items():
        model_input_cost = (total_input_tokens / 1000) * prices['input']
        model_output_cost = (total_output_tokens / 1000) * prices['output']
        model_total = model_input_cost + model_output_cost
        
        comparison_data.append({
            'Model': model,
            'Monthly Cost': f"${model_total:.2f}",
            'Cost per Query': f"${model_total/queries_per_month:.4f}",
            'Annual Cost': f"${model_total*12:.2f}"
        })
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

def main():
    """Main application"""
    st.title("📚 Enhanced Multi-Format RAG System")
    st.markdown("Process and query documents across multiple formats: PDF, Word, Excel, PowerPoint")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ System Status")
        
        # API Key status
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success(f"✅ API Key: {api_key[:10]}...{api_key[-4:]}")
        else:
            st.warning("⚠️ No API key found")
            new_key = st.text_input("Enter OpenAI API Key", type="password")
            if new_key:
                os.environ["OPENAI_API_KEY"] = new_key
                st.success("✅ API key updated!")
                st.experimental_rerun()
        
        # System statistics
        if st.session_state.rag_system:
            try:
                stats = st.session_state.rag_system.get_database_stats()
                st.metric("Database Chunks", stats.get('total_chunks', 0))
                st.metric("Processed Files", len(st.session_state.processed_files))
            except:
                pass
        
        # Reset button
        if st.button("🔄 Reset System"):
            st.session_state.rag_system = None
            st.session_state.processed_files = []
            st.session_state.processing_stats = {}
            st.success("System reset!")
            st.experimental_rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Document Selection", "📤 File Upload", "🔍 Query Documents", "💰 Cost Estimator"])
    
    with tab1:
        display_file_selection()
    
    with tab2:
        display_file_upload_interface()
    
    with tab3:
        display_query_interface()
    
    with tab4:
        display_cost_estimator()

if __name__ == "__main__":
    main()