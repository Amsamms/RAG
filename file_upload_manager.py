#!/usr/bin/env python3
"""
File Upload Manager for Streamlit RAG System
Handles file uploads and management
"""

import streamlit as st
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
import tempfile

class FileUploadManager:
    def __init__(self, upload_directory: str = "./uploaded_docs"):
        self.upload_directory = Path(upload_directory)
        self.upload_directory.mkdir(exist_ok=True)
        
        # Supported file types
        self.supported_types = {
            'PDF': ['pdf'],
            'Word': ['docx', 'doc'],
            'Excel': ['xlsx', 'xls'],
            'PowerPoint': ['pptx', 'ppt'],
            'Text': ['txt']
        }
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions"""
        extensions = []
        for category, exts in self.supported_types.items():
            extensions.extend(exts)
        return extensions
    
    def save_uploaded_file(self, uploaded_file, category: str = None) -> Dict[str, Any]:
        """Save uploaded file to disk and return file info"""
        try:
            # Create category subdirectory if specified
            if category:
                save_dir = self.upload_directory / category.lower()
                save_dir.mkdir(exist_ok=True)
            else:
                save_dir = self.upload_directory
            
            # Save file
            file_path = save_dir / uploaded_file.name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Get file info
            file_stat = file_path.stat()
            file_info = {
                'name': uploaded_file.name,
                'path': str(file_path),
                'size': file_stat.st_size,
                'type': uploaded_file.type,
                'category': category or self.get_file_category(uploaded_file.name),
                'status': 'uploaded'
            }
            
            return file_info
            
        except Exception as e:
            return {'error': str(e), 'status': 'failed'}
    
    def get_file_category(self, filename: str) -> str:
        """Determine file category from filename"""
        extension = Path(filename).suffix.lower().lstrip('.')
        
        for category, extensions in self.supported_types.items():
            if extension in extensions:
                return category
        
        return 'Unknown'
    
    def list_uploaded_files(self) -> Dict[str, List[Dict]]:
        """List all uploaded files by category"""
        files_by_category = {}
        
        for category in self.supported_types.keys():
            files_by_category[category] = []
            category_dir = self.upload_directory / category.lower()
            
            if category_dir.exists():
                for file_path in category_dir.iterdir():
                    if file_path.is_file():
                        file_stat = file_path.stat()
                        file_info = {
                            'name': file_path.name,
                            'path': str(file_path),
                            'size': file_stat.st_size,
                            'category': category,
                            'extension': file_path.suffix.lower()
                        }
                        files_by_category[category].append(file_info)
        
        return files_by_category
    
    def delete_file(self, file_path: str) -> bool:
        """Delete uploaded file"""
        try:
            Path(file_path).unlink()
            return True
        except Exception:
            return False
    
    def clear_category(self, category: str) -> int:
        """Clear all files in a category"""
        category_dir = self.upload_directory / category.lower()
        deleted_count = 0
        
        if category_dir.exists():
            for file_path in category_dir.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception:
                        pass
        
        return deleted_count

def display_file_upload_interface():
    """Display file upload interface in Streamlit"""
    st.header("📤 File Upload Manager")
    
    upload_manager = FileUploadManager()
    
    # Upload section
    st.subheader("📁 Upload New Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=upload_manager.get_supported_extensions(),
        accept_multiple_files=True,
        help="Supported formats: PDF, Word, Excel, PowerPoint, Text"
    )
    
    if uploaded_files:
        st.success(f"Selected {len(uploaded_files)} files for upload")
        
        # Upload options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            organize_by_type = st.checkbox("Organize files by type", value=True)
        
        with col2:
            if st.button("📤 Upload Files", type="primary"):
                upload_files(uploaded_files, upload_manager, organize_by_type)
    
    # Display uploaded files
    st.subheader("📋 Uploaded Documents")
    
    uploaded_files_by_category = upload_manager.list_uploaded_files()
    
    for category, files in uploaded_files_by_category.items():
        if files:
            with st.expander(f"📊 {category} Files ({len(files)} files)", expanded=True):
                
                # Category controls
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button(f"🗑️ Clear All {category}", key=f"clear_{category}"):
                        deleted = upload_manager.clear_category(category)
                        st.success(f"Deleted {deleted} {category} files")
                        st.experimental_rerun()
                
                # File list
                for file_info in files:
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.text(file_info['name'])
                    
                    with col2:
                        st.text(format_file_size(file_info['size']))
                    
                    with col3:
                        st.text(file_info['extension'])
                    
                    with col4:
                        if st.button("🗑️", key=f"delete_{file_info['path']}"):
                            if upload_manager.delete_file(file_info['path']):
                                st.success(f"Deleted {file_info['name']}")
                                st.experimental_rerun()

def upload_files(uploaded_files, upload_manager, organize_by_type):
    """Handle file upload process"""
    success_count = 0
    error_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Uploading: {uploaded_file.name}")
        
        # Determine category
        category = upload_manager.get_file_category(uploaded_file.name) if organize_by_type else None
        
        # Save file
        result = upload_manager.save_uploaded_file(uploaded_file, category)
        
        if result.get('status') == 'uploaded':
            success_count += 1
        else:
            error_count += 1
            st.error(f"Failed to upload {uploaded_file.name}: {result.get('error', 'Unknown error')}")
    
    # Final status
    progress_bar.progress(1.0)
    status_text.text("Upload complete!")
    
    if success_count > 0:
        st.success(f"Successfully uploaded {success_count} files")
    if error_count > 0:
        st.error(f"Failed to upload {error_count} files")

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

if __name__ == "__main__":
    # Test the upload manager
    display_file_upload_interface()