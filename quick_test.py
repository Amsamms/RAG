#!/usr/bin/env python3
"""
Quick test for search accuracy and API key
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file

from enhanced_rag_system import EnhancedPDFRAGSystem

def main():
    print("🔧 Quick RAG Test")
    print("="*50)
    
    # Test 1: Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"🔑 API Key: {'✅ Found' if api_key else '❌ Missing'}")
    if api_key:
        print(f"   Key: {api_key[:10]}...{api_key[-10:]}")
    
    # Test 2: Initialize RAG
    print("\n🚀 Initializing RAG...")
    rag = EnhancedPDFRAGSystem()
    
    # Test 3: Check database
    stats = rag.get_database_stats()
    print(f"📊 Database: {stats['total_chunks']} chunks")
    print(f"🤖 LLM: {'✅ Available' if stats['llm_available'] else '❌ Not Available'}")
    
    # Test 4: Process PDFs if needed
    if stats['total_chunks'] == 0:
        print("\n📚 Processing PDFs...")
        rag.process_pdfs(".")
        stats = rag.get_database_stats()
        print(f"✅ Now have {stats['total_chunks']} chunks")
    
    # Test 5: Search accuracy test
    print(f"\n🔍 Testing search for 'ethanolamine':")
    
    # Direct search first
    import fitz
    doc = fitz.open("LPG MEROX GOM.unlocked.pdf")
    direct_pages = []
    for page_num in range(len(doc)):
        text = doc.load_page(page_num).get_text().lower()
        if 'ethanolamine' in text:
            direct_pages.append(page_num + 1)
    doc.close()
    
    print(f"📄 Direct search: Found on pages {direct_pages}")
    
    # Vector search
    results = rag.search_documents("ethanolamine", n_results=5)
    print(f"🧮 Vector search: {len(results['results'])} results")
    for i, r in enumerate(results['results'][:3], 1):
        print(f"   {i}. {r['document']} page {r['page']} (distance: {r.get('distance', 0):.3f})")
    
    # Test 6: LLM response
    if stats['llm_available']:
        print(f"\n🤖 Testing LLM response...")
        response = rag.ask_question("where is ethanolamine mentioned?", use_llm=True, n_results=3)
        if response['llm_response']:
            print(f"✅ LLM Response: {response['llm_response'][:100]}...")
        else:
            print("❌ No LLM response")
    else:
        print(f"\n❌ LLM not available")
    
    print(f"\n" + "="*50)
    print("✅ Test complete!")

if __name__ == "__main__":
    main()