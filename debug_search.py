#!/usr/bin/env python3
"""
Debug script to test search accuracy and API issues
"""

from enhanced_rag_system import EnhancedPDFRAGSystem
import fitz
import os

def test_direct_text_search(query="ethanolamine"):
    """Test direct text search to compare with vector search"""
    print(f"🔍 Direct text search for '{query}':")
    print("="*60)
    
    pdf_files = ["CCR Cycle Max GOM.unlocked.pdf", "LPG MEROX GOM.unlocked.pdf", "PENEX H.O.T GOM.pdf"]
    
    for pdf_file in pdf_files:
        try:
            doc = fitz.open(pdf_file)
            found_pages = []
            contexts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text().lower()
                
                if query.lower() in text:
                    found_pages.append(page_num + 1)
                    # Get context around the word
                    lines = text.split('\n')
                    for line in lines:
                        if query.lower() in line:
                            contexts.append(line.strip()[:100] + "...")
            
            doc.close()
            
            if found_pages:
                print(f"📄 {pdf_file}: Found on pages {found_pages}")
                print(f"   Contexts: {contexts[:2]}")  # Show first 2 contexts
            else:
                print(f"📄 {pdf_file}: Not found")
                
        except Exception as e:
            print(f"📄 {pdf_file}: Error - {e}")
    
    print()

def test_vector_search():
    """Test vector search accuracy"""
    print("🤖 Testing Vector Search:")
    print("="*60)
    
    # Initialize without API key first
    old_key = os.environ.get("OPENAI_API_KEY")
    if old_key:
        del os.environ["OPENAI_API_KEY"]
    
    try:
        rag = EnhancedPDFRAGSystem()
        
        # Get database stats
        stats = rag.get_database_stats()
        print(f"📊 Database stats: {stats}")
        
        # If no chunks, process PDFs
        if stats.get('total_chunks', 0) == 0:
            print("\n📚 Processing PDFs...")
            chunks = rag.process_pdfs(".")
            print(f"✅ Processed {chunks} chunks")
        
        # Test different queries
        test_queries = [
            "ethanolamine",
            "ethanolamine mentioned",
            "amine",
            "chemical compounds",
            "acid gas removal",
            "methyldiethanolamine MDEA"
        ]
        
        print(f"\n🔍 Testing {len(test_queries)} queries:")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = rag.search_documents(query, n_results=3)
            
            if results['results']:
                for i, result in enumerate(results['results'], 1):
                    print(f"  {i}. {result['document']} (page {result['page']}) - Distance: {result.get('distance', 'N/A'):.3f}")
                    print(f"     Text: {result['text'][:80]}...")
            else:
                print("  ❌ No results found")
    
    except Exception as e:
        print(f"❌ Error in vector search: {e}")
    
    finally:
        # Restore API key
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key

def test_api_key():
    """Test API key functionality"""
    print("🔑 Testing API Key:")
    print("="*60)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found in environment")
        print("💡 Solutions:")
        print("   1. Create .env file: cp .env.template .env")
        print("   2. Edit .env and add: OPENAI_API_KEY=your_key_here")
        print("   3. Or set in terminal: export OPENAI_API_KEY=your_key")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test API key validity
    try:
        import openai
        openai.api_key = api_key
        
        # Simple test call
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print("✅ API key is valid and working")
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        print("💡 Check if:")
        print("   1. API key is correct")
        print("   2. You have credits in your OpenAI account")
        print("   3. API key has proper permissions")
        return False

def suggest_improvements():
    """Suggest improvements for search accuracy"""
    print("💡 Improving Search Accuracy:")
    print("="*60)
    print("1. **Exact Word Search**: Use direct text search for specific terms")
    print("2. **Semantic Search**: Use vector search for concept queries")
    print("3. **Hybrid Approach**: Combine both methods")
    print("4. **Chunk Size**: Smaller chunks = more precise but less context")
    print("5. **More Results**: Increase n_results to see more options")
    print("6. **Different Embeddings**: Try different embedding models")

def main():
    print("🔧 RAG System Debug Tool")
    print("="*80)
    
    # Test 1: Direct text search
    test_direct_text_search("ethanolamine")
    
    # Test 2: Vector search
    test_vector_search()
    
    # Test 3: API key
    test_api_key()
    
    # Test 4: Suggestions
    suggest_improvements()

if __name__ == "__main__":
    main()