#!/usr/bin/env python3
"""
Secure Streamlit App - API Keys Only Through User Input
No API keys stored in code or environment files
"""

import streamlit as st
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any
import time
from datetime import datetime
from dotenv import load_dotenv

# Import our secure system
from secure_rag_system import SecureMultiFormatRAG
from file_upload_manager import FileUploadManager, display_file_upload_interface

# Load environment variables (no API keys)
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Secure RAG System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.security-warning {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
.model-card {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "gpt-3.5-turbo"
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

def initialize_rag_system():
    """Initialize the secure RAG system"""
    if st.session_state.rag_system is None:
        with st.spinner("🔒 Initializing secure RAG system..."):
            st.session_state.rag_system = SecureMultiFormatRAG()
    return st.session_state.rag_system

def display_api_key_configuration():
    """Display secure API key configuration"""
    st.header("🔑 Secure API Configuration")
    
    # Security notice
    st.markdown("""
    <div class="security-warning">
    <strong>🔒 Security Notice:</strong> Your API key is only stored in memory during this session. 
    It is never saved to disk or environment files. You will need to re-enter it if you refresh the page.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    rag = initialize_rag_system()
    
    # API Key input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        api_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            help="Your API key starts with 'sk-' and is never stored on disk",
            placeholder="sk-..."
        )
    
    with col2:
        # Current status
        if st.session_state.api_key_set:
            st.success("🔑 API Key Set")
        else:
            st.warning("❌ No API Key")
    
    # Model selection
    st.subheader("🤖 Model Selection")
    
    available_models = rag.get_available_models()
    
    # Display model options in cards
    cols = st.columns(3)
    model_keys = list(available_models.keys())
    
    for i, (model_id, model_info) in enumerate(available_models.items()):
        col_idx = i % 3
        with cols[col_idx]:
            st.markdown(f"""
            <div class="model-card">
                <h4>{model_info['name']}</h4>
                <p><strong>Cost:</strong> ${model_info['cost_per_1k_input']:.4f} input / ${model_info['cost_per_1k_output']:.4f} output per 1K tokens</p>
                <p><strong>Best for:</strong> {model_info['recommended_for']}</p>
                <p><small>{model_info['description']}</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Select {model_info['name']}", key=f"select_{model_id}"):
                st.session_state.selected_model = model_id
                st.success(f"Selected: {model_info['name']}")
    
    # Show current selection
    current_model = available_models.get(st.session_state.selected_model, {})
    st.info(f"🎯 Currently selected: **{current_model.get('name', 'None')}** ({st.session_state.selected_model})")
    
    # Set credentials button
    if st.button("🔒 Set API Key & Model", type="primary"):
        if api_key and api_key.strip():
            success = rag.set_openai_credentials(api_key, st.session_state.selected_model)
            if success:
                st.session_state.api_key_set = True
                st.success(f"✅ API key set successfully with model: {st.session_state.selected_model}")
                
                # Show cost estimate
                cost_info = rag.calculate_model_cost(st.session_state.selected_model, 100)
                st.info(f"💰 Estimated cost for 100 queries: ${cost_info['monthly_cost']}")
                st.rerun()
            else:
                st.error("❌ Invalid API key format. Make sure it starts with 'sk-'")
        else:
            st.error("❌ Please enter your API key")
    
    return rag

def display_cost_calculator():
    """Display detailed cost calculator"""
    st.header("💰 Cost Calculator")
    
    rag = st.session_state.rag_system
    if not rag:
        st.error("Please initialize the system first")
        return
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        queries_per_month = st.number_input("Queries per month", 1, 100000, 100)
        avg_input_tokens = st.slider("Average input tokens (context + question)", 500, 8000, 2000)
    
    with col2:
        avg_output_tokens = st.slider("Average output tokens (response length)", 50, 1000, 200)
        time_period = st.selectbox("Time period", ["Monthly", "Annual"])
    
    # Calculate costs for all models
    st.subheader("📊 Cost Comparison Across All Models")
    
    cost_data = []
    available_models = rag.get_available_models()
    
    for model_id, model_info in available_models.items():
        cost_info = rag.calculate_model_cost(model_id, queries_per_month, avg_input_tokens, avg_output_tokens)
        
        if time_period == "Annual":
            display_cost = cost_info['annual_cost']
            cost_per_query = cost_info['cost_per_query']
        else:
            display_cost = cost_info['monthly_cost']
            cost_per_query = cost_info['cost_per_query']
        
        cost_data.append({
            'Model': model_info['name'],
            f'{time_period} Cost': f"${display_cost}",
            'Cost per Query': f"${cost_per_query:.6f}",
            'Input Cost/1K': f"${model_info['cost_per_1k_input']:.4f}",
            'Output Cost/1K': f"${model_info['cost_per_1k_output']:.4f}",
            'Recommended For': model_info['recommended_for']
        })
    
    # Display as table
    df = pd.DataFrame(cost_data)
    st.dataframe(df, use_container_width=True)
    
    # Highlight current selection
    if st.session_state.selected_model in available_models:
        current_model = available_models[st.session_state.selected_model]
        current_cost = rag.calculate_model_cost(st.session_state.selected_model, queries_per_month, avg_input_tokens, avg_output_tokens)
        
        st.subheader(f"🎯 Your Selected Model: {current_model['name']}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            period_cost = current_cost['annual_cost'] if time_period == "Annual" else current_cost['monthly_cost']
            st.metric(f"{time_period} Cost", f"${period_cost}")
        with col2:
            st.metric("Per Query", f"${current_cost['cost_per_query']:.6f}")
        with col3:
            st.metric("Input Cost", f"${current_cost['input_cost']:.4f}")
        with col4:
            st.metric("Output Cost", f"${current_cost['output_cost']:.4f}")

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

def display_document_interface():
    """Display comprehensive document processing interface with multi-format support"""
    st.header("📁 Multi-Format Document Processing")
    
    rag = st.session_state.rag_system
    if not rag:
        st.error("System not initialized")
        return
    
    # Tab for different document management approaches
    doc_tab1, doc_tab2 = st.tabs(["📂 Select Local Files", "📤 Upload New Files"])
    
    with doc_tab1:
        st.subheader("📋 Available Documents by Category")
        
        # Get file information
        file_categories = get_file_info(".")
        selected_files = []
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
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
                                value=min(5, len(info['files'])),  # Default to 5 or less
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
                    st.info(f"No {category} files found")
        
        with col2:
            st.subheader("📊 Selection Summary")
            
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
                    process_selected_files(selected_files, chunk_size, max_workers, batch_size, rag)
            else:
                st.info("Select documents to see processing options")
    
    with doc_tab2:
        # File upload interface
        st.subheader("📤 Upload New Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'pptx', 'ppt', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, Word, Excel, PowerPoint, Text"
        )
        
        if uploaded_files:
            st.success(f"Selected {len(uploaded_files)} files for upload")
            
            # Upload and process options
            col1, col2 = st.columns([2, 1])
            
            with col1:
                organize_by_type = st.checkbox("Organize files by type", value=True)
                process_immediately = st.checkbox("Process immediately after upload", value=True)
            
            with col2:
                if st.button("📤 Upload & Process Files", type="primary"):
                    upload_and_process_files(uploaded_files, organize_by_type, process_immediately, rag)

def process_selected_files(selected_files: List[Dict], chunk_size: int, max_workers: int, batch_size: int, rag):
    """Process the selected files with full multi-format support"""
    if not selected_files:
        st.error("No files selected for processing")
        return
    
    # Processing progress
    progress_container = st.container()
    
    with progress_container:
        st.subheader("🔄 Processing Multi-Format Documents")
        
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
            status_text.text(f"Processing: {file_info['name']} ({i+1}/{len(selected_files)}) - {file_info['category']}")
            
            try:
                # Process single file using the secure RAG system
                file_chunks = rag.process_single_file(Path(file_info['path']))
                
                if file_chunks:
                    # Add to database
                    documents = []
                    embeddings = []
                    metadatas = []
                    ids = []
                    
                    for chunk_data in file_chunks:
                        # Create embeddings
                        embedding = rag.embedding_model.encode(chunk_data['text'])
                        
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
                    
                    # Add to database in batches
                    for j in range(0, len(documents), batch_size):
                        batch_docs = documents[j:j+batch_size]
                        batch_embeddings = embeddings[j:j+batch_size]
                        batch_metadatas = metadatas[j:j+batch_size]
                        batch_ids = ids[j:j+batch_size]
                        
                        rag.collection.add(
                            documents=batch_docs,
                            embeddings=batch_embeddings,
                            metadatas=batch_metadatas,
                            ids=batch_ids
                        )
                    
                    total_chunks += len(file_chunks)
                    processing_stats['successful'] += 1
                    
                    # Update category stats
                    category = file_info['category']
                    if category not in processing_stats['categories']:
                        processing_stats['categories'][category] = {'files': 0, 'chunks': 0}
                    processing_stats['categories'][category]['files'] += 1
                    processing_stats['categories'][category]['chunks'] += len(file_chunks)
                    
                    st.success(f"✅ {file_info['category']}: {file_info['name']} → {len(file_chunks)} chunks")
                    
                else:
                    processing_stats['failed'] += 1
                    st.warning(f"⚠️ No content extracted from {file_info['name']}")
                    
            except Exception as e:
                processing_stats['failed'] += 1
                st.error(f"❌ Error processing {file_info['name']}: {str(e)}")
        
        # Complete processing
        processing_stats['total_chunks'] = total_chunks
        processing_stats['processing_time'] = time.time() - start_time
        
        # Update session state
        st.session_state.processed_files.extend(selected_files)
        
        # Show results
        progress_bar.progress(1.0)
        status_text.text("✅ Multi-Format Processing Complete!")
        
        # Display results
        st.success(f"🎉 Successfully processed {processing_stats['successful']} files across multiple formats!")
        st.info(f"📊 Total chunks created: {processing_stats['total_chunks']}")
        st.info(f"⏱️ Processing time: {processing_stats['processing_time']:.2f} seconds")
        
        # Category breakdown
        if processing_stats['categories']:
            st.subheader("📋 Multi-Format Processing Summary")
            cols = st.columns(len(processing_stats['categories']))
            for i, (category, stats) in enumerate(processing_stats['categories'].items()):
                with cols[i]:
                    st.metric(
                        f"📊 {category}",
                        f"{stats['files']} files",
                        f"{stats['chunks']} chunks"
                    )

def upload_and_process_files(uploaded_files, organize_by_type, process_immediately, rag):
    """Handle file upload and processing"""
    upload_manager = FileUploadManager()
    
    success_count = 0
    uploaded_file_info = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Upload files
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Uploading: {uploaded_file.name}")
        
        # Determine category
        category = upload_manager.get_file_category(uploaded_file.name) if organize_by_type else None
        
        # Save file
        result = upload_manager.save_uploaded_file(uploaded_file, category)
        
        if result.get('status') == 'uploaded':
            success_count += 1
            uploaded_file_info.append({
                'name': uploaded_file.name,
                'path': result['path'],
                'category': result['category'],
                'size': result['size'],
                'extension': Path(uploaded_file.name).suffix
            })
        else:
            st.error(f"Failed to upload {uploaded_file.name}: {result.get('error', 'Unknown error')}")
    
    progress_bar.progress(1.0)
    status_text.text("Upload complete!")
    
    if success_count > 0:
        st.success(f"Successfully uploaded {success_count} files")
        
        # Process immediately if requested
        if process_immediately and uploaded_file_info:
            st.info("🔄 Processing uploaded files...")
            process_selected_files(uploaded_file_info, 500, 4, 50, rag)
    
    if len(uploaded_files) - success_count > 0:
        st.error(f"Failed to upload {len(uploaded_files) - success_count} files")

def display_query_interface():
    """Display query interface"""
    if not st.session_state.api_key_set:
        st.warning("🔑 Please set your API key in the API Configuration tab first")
        return
    
    st.header("🔍 Query Documents")
    
    rag = st.session_state.rag_system
    if not rag:
        st.error("System not initialized")
        return
    
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
        file_type_filter = st.selectbox(
            "Filter by File Type",
            ["All Files", "PDF", "Word", "Excel", "PowerPoint"]
        )
    
    # Search button
    if st.button("🔍 Search", type="primary") and question:
        if not rag.llm_available and use_llm:
            st.error("❌ LLM not available. Please check your API key.")
            return
        
        with st.spinner("🔍 Searching documents..."):
            try:
                # Perform search
                results = rag.search_documents(question, n_results)
                
                # Filter by file type if specified
                if file_type_filter != "All Files":
                    filtered_results = []
                    for result in results['results']:
                        result_file_type = result.get('file_type', '').lower()
                        filter_type = file_type_filter.lower()
                        
                        # Map filter names to actual file types
                        type_mapping = {
                            'pdf': 'pdf',
                            'word': 'word',
                            'excel': 'excel', 
                            'powerpoint': 'powerpoint'
                        }
                        
                        if result_file_type == type_mapping.get(filter_type, filter_type):
                            filtered_results.append(result)
                    
                    results['results'] = filtered_results
                    st.info(f"🔍 Filtered to {file_type_filter} files only: {len(filtered_results)} results")
                
                # Display results
                if results['results']:
                    if use_llm and rag.llm_available:
                        st.subheader("🤖 AI Response")
                        try:
                            ai_response = rag.generate_llm_response(
                                question, results['results'][:5], st.session_state.selected_model
                            )
                            st.info(ai_response)
                        except Exception as e:
                            st.error(f"❌ AI response failed: {str(e)}")
                            st.info("This might be due to an invalid API key or insufficient credits.")
                    
                    # Search Results
                    st.subheader(f"📋 Search Results ({len(results['results'])} found)")
                    
                    # Group results by file type for better organization
                    results_by_type = {}
                    for result in results['results']:
                        file_type = result.get('file_type', 'unknown').title()
                        if file_type not in results_by_type:
                            results_by_type[file_type] = []
                        results_by_type[file_type].append(result)
                    
                    # Display results grouped by file type
                    for file_type, type_results in results_by_type.items():
                        st.markdown(f"### 📊 {file_type} Files ({len(type_results)} results)")
                        
                        for i, result in enumerate(type_results, 1):
                            file_type_emoji = {
                                'Pdf': '📄',
                                'Word': '📝', 
                                'Excel': '📊',
                                'Powerpoint': '📈',
                                'Unknown': '📋'
                            }.get(file_type, '📋')
                            
                            with st.expander(f"{file_type_emoji} Result {i}: {result['document']} (Page {result['page']}) - Similarity: {1-result.get('distance', 0):.3f}"):
                                st.text(result['text'])
                                
                                # Enhanced metadata display
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.caption(f"📄 Document: {result['document']}")
                                with col2:
                                    st.caption(f"📄 Page: {result['page']}")
                                with col3:
                                    st.caption(f"📊 Type: {result.get('file_type', 'unknown').title()}")
                                with col4:
                                    st.caption(f"🎯 Relevance: {1-result.get('distance', 0):.3f}")
                else:
                    if file_type_filter != "All Files":
                        st.warning(f"No relevant {file_type_filter} documents found for your query")
                    else:
                        st.warning("No relevant documents found for your query")
                    
            except Exception as e:
                st.error(f"Search error: {str(e)}")

def main():
    """Main application"""
    st.title("🔒 Secure Multi-Format RAG System")
    st.markdown("**No API keys stored in code or files - Your data stays secure!**")
    
    # Sidebar with system status
    with st.sidebar:
        st.header("🛠️ System Status")
        
        # API Key Status
        if st.session_state.api_key_set:
            st.success("🔑 API Key: ✅ Set")
            st.info(f"🤖 Model: {st.session_state.selected_model}")
        else:
            st.warning("🔑 API Key: ❌ Not Set")
        
        # Reset button
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session reset!")
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔑 API Configuration", 
        "📁 Documents", 
        "🔍 Query", 
        "💰 Cost Calculator"
    ])
    
    with tab1:
        display_api_key_configuration()
    
    with tab2:
        display_document_interface()
    
    with tab3:
        display_query_interface()
    
    with tab4:
        display_cost_calculator()

if __name__ == "__main__":
    main()