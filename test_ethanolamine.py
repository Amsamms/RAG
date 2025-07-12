#!/usr/bin/env python3
"""
Test script to specifically search for ethanolamine mentions
"""

from rag_system import PDFRAGSystem
import fitz

def direct_text_search():
    """Direct text search for ethanolamine in PDFs"""
    print("🔍 Direct text search for 'ethanolamine':")
    print("="*60)
    
    pdf_files = ["CCR Cycle Max GOM.unlocked.pdf", "LPG MEROX GOM.unlocked.pdf", "PENEX H.O.T GOM.pdf"]
    
    for pdf_file in pdf_files:
        try:
            doc = fitz.open(pdf_file)
            found_pages = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text().lower()
                
                if 'ethanolamine' in text:
                    found_pages.append(page_num + 1)
            
            doc.close()
            
            if found_pages:
                print(f"📄 {pdf_file}: Found on pages {found_pages}")
            else:
                print(f"📄 {pdf_file}: Not found")
                
        except Exception as e:
            print(f"📄 {pdf_file}: Error - {e}")
    
    print()

def rag_search():
    """RAG-based semantic search"""
    print("🤖 RAG-based semantic search:")
    print("="*60)
    
    # Initialize RAG system
    rag = PDFRAGSystem()
    
    # Different query variations
    queries = [
        "ethanolamine",
        "ethanolamine mentioned",
        "where is ethanolamine",
        "amine compounds",
        "chemical amine"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = rag.query(query, n_results=3)
        
        if results['results']:
            for i, result in enumerate(results['results'], 1):
                print(f"  {i}. {result['document']} (page {result['page']})")
                # Show snippet of text
                text_snippet = result['text'][:100].replace('\n', ' ')
                print(f"     Text: {text_snippet}...")
        else:
            print("  No results found")

def main():
    print("🚀 Ethanolamine Search Test")
    print("="*60)
    
    # First do direct text search
    direct_text_search()
    
    # Then do RAG search
    rag_search()

if __name__ == "__main__":
    main()