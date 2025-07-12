#!/usr/bin/env python3
"""
Explanation and demonstration of where chunks and vectors are stored
"""

import chromadb
import os
from pathlib import Path
import json

def explain_chromadb_storage():
    """Explain ChromaDB storage mechanism"""
    print("🗄️ ChromaDB Storage Explanation")
    print("=" * 60)
    
    # Initialize ChromaDB client
    client = chromadb.Client()
    
    print("\n📍 Current Storage Mode: IN-MEMORY")
    print("   • Vectors and chunks stored in RAM only")
    print("   • Data lost when program/session ends")
    print("   • No files created on disk")
    print("   • Fast but temporary")
    
    # Check for any existing collections
    try:
        collections = client.list_collections()
        print(f"\n📊 Active Collections: {len(collections)}")
        for collection in collections:
            stats = collection.count()
            print(f"   • {collection.name}: {stats} chunks")
    except Exception as e:
        print(f"   Error checking collections: {e}")
    
    print("\n🔧 Alternative Storage Options:")
    print("   1. Persistent SQLite (local file)")
    print("   2. Persistent PostgreSQL (database)")
    print("   3. Remote ChromaDB server")
    
    return client

def show_persistent_storage_example():
    """Show how to enable persistent storage"""
    print("\n💾 Persistent Storage Example")
    print("=" * 40)
    
    # Show current temporary storage
    print("🔸 Current (Temporary):")
    print("   client = chromadb.Client()  # In-memory only")
    
    print("\n🔸 Persistent Storage Option:")
    print("   client = chromadb.PersistentClient(path='./chroma_db')")
    print("   # Creates ./chroma_db/ directory with SQLite files")
    
    print("\n📁 Persistent storage would create:")
    print("   ./chroma_db/")
    print("   ├── chroma.sqlite3       # Main database")
    print("   ├── index/              # Vector indices") 
    print("   └── collections/        # Collection data")

def show_current_data_flow():
    """Explain the current data flow"""
    print("\n🔄 Current Data Flow")
    print("=" * 40)
    
    print("1️⃣ Document Processing:")
    print("   PDF/Word/Excel → Text Extraction → Text Chunks")
    
    print("\n2️⃣ Vector Generation:")
    print("   Text Chunks → SentenceTransformer → 384-dim vectors")
    
    print("\n3️⃣ Storage (IN-MEMORY):")
    print("   Vectors + Metadata → ChromaDB Collection (RAM)")
    print("   └── Collection Name: 'pdf_documents' or 'secure_documents'")
    print("   └── No disk files created")
    
    print("\n4️⃣ Search:")
    print("   Query → Vector → Similarity Search → Results")
    
    print("\n⚠️ Data Persistence:")
    print("   • Lost when Streamlit app is restarted")
    print("   • Lost when terminal is closed")
    print("   • Must re-process documents each session")

def check_actual_storage_location():
    """Check where ChromaDB actually stores data"""
    print("\n🔍 Checking Actual Storage Location")
    print("=" * 45)
    
    # Check default ChromaDB settings
    try:
        import chromadb
        client = chromadb.Client()
        
        print("📍 ChromaDB Client Type:", type(client).__name__)
        print("📍 Storage Mode: In-Memory (EphemeralClient)")
        print("📍 No persistent files created")
        
        # Check for any chroma directories
        possible_locations = [
            "./chroma",
            "./chroma_db", 
            "./chroma.db",
            "~/.chroma",
            "/tmp/chroma"
        ]
        
        print("\n🔍 Checking for ChromaDB directories:")
        for location in possible_locations:
            path = Path(location).expanduser()
            if path.exists():
                print(f"   ✅ Found: {path}")
                if path.is_dir():
                    files = list(path.iterdir())
                    print(f"      Files: {[f.name for f in files]}")
            else:
                print(f"   ❌ Not found: {location}")
                
    except Exception as e:
        print(f"Error checking storage: {e}")

def show_memory_usage():
    """Show approximate memory usage"""
    print("\n💾 Memory Usage Estimation")
    print("=" * 35)
    
    print("📊 Per Text Chunk:")
    print("   • Text content: ~500 characters = ~500 bytes")
    print("   • Vector (384 dimensions): 384 × 4 bytes = 1,536 bytes")  
    print("   • Metadata: ~200 bytes")
    print("   • Total per chunk: ~2,236 bytes (≈2.2 KB)")
    
    print("\n📈 Scaling Examples:")
    examples = [
        (100, "Small document"),
        (1000, "Medium document set"),
        (10000, "Large document set"),
        (50000, "Enterprise scale")
    ]
    
    for chunks, description in examples:
        memory_kb = chunks * 2.2
        memory_mb = memory_kb / 1024
        print(f"   • {chunks:,} chunks ({description}): {memory_mb:.1f} MB")

def main():
    """Main explanation function"""
    print("🧠 RAG System Storage Deep Dive")
    print("=" * 70)
    
    # Explain current storage
    client = explain_chromadb_storage()
    
    # Show persistent alternatives
    show_persistent_storage_example()
    
    # Explain data flow
    show_current_data_flow()
    
    # Check actual locations
    check_actual_storage_location()
    
    # Show memory usage
    show_memory_usage()
    
    print("\n" + "=" * 70)
    print("🎯 Summary:")
    print("   • Current: IN-MEMORY storage (temporary)")
    print("   • Location: RAM only, no disk files")
    print("   • Benefit: Fast access, no cleanup needed")
    print("   • Drawback: Data lost on restart")
    print("   • Solution: Enable persistent storage if needed")
    print("=" * 70)

if __name__ == "__main__":
    main()