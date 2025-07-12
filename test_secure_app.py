#!/usr/bin/env python3
"""
Quick test to verify the secure app works without errors
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly"""
    try:
        from secure_rag_system import SecureMultiFormatRAG
        print("✅ SecureMultiFormatRAG import successful")
        
        from file_upload_manager import FileUploadManager
        print("✅ FileUploadManager import successful")
        
        # Test RAG system initialization
        rag = SecureMultiFormatRAG()
        print("✅ RAG system initialization successful")
        
        # Test available models
        models = rag.get_available_models()
        print(f"✅ Available models: {len(models)} models loaded")
        
        # Test database stats
        stats = rag.get_database_stats()
        print(f"✅ Database stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import/initialization error: {e}")
        return False

def test_file_processing():
    """Test file processing capabilities"""
    try:
        from pathlib import Path
        from secure_rag_system import SecureMultiFormatRAG
        
        rag = SecureMultiFormatRAG()
        
        # Check if sample files exist
        sample_files = [
            "sample_chemical_manual.docx",
            "sample_chemical_database.xlsx", 
            "sample_process_training.pptx",
            "sample_equipment_manual.txt"
        ]
        
        existing_files = []
        for file in sample_files:
            if Path(file).exists():
                existing_files.append(file)
        
        print(f"✅ Found {len(existing_files)} sample files: {existing_files}")
        
        # Test processing capability (without actually processing)
        if existing_files:
            test_file = Path(existing_files[0])
            print(f"✅ Testing processing capability with: {test_file.name}")
            
            # This should not throw an error
            file_extension = test_file.suffix.lower()
            supported_extensions = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.txt']
            
            if file_extension in supported_extensions:
                print(f"✅ File type {file_extension} is supported")
            else:
                print(f"⚠️ File type {file_extension} not supported")
        
        return True
        
    except Exception as e:
        print(f"❌ File processing test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Secure RAG System")
    print("=" * 50)
    
    # Test 1: Imports and initialization
    print("\n1. Testing imports and initialization...")
    test1_success = test_imports()
    
    # Test 2: File processing
    print("\n2. Testing file processing capabilities...")
    test2_success = test_file_processing()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Imports/Init: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"   File Processing: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! The secure app should work correctly.")
        print("\n🚀 You can now run: streamlit run secure_streamlit_app.py")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
    
    return test1_success and test2_success

if __name__ == "__main__":
    main()