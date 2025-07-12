#!/usr/bin/env python3
"""
Hybrid search system combining exact text search and semantic search
"""

import fitz
import os
from pathlib import Path
from enhanced_rag_system import EnhancedPDFRAGSystem
from typing import List, Dict, Any

class HybridSearchRAG:
    def __init__(self):
        self.rag_system = EnhancedPDFRAGSystem()
        self.pdf_files = list(Path(".").glob("*.pdf"))
    
    def exact_text_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform exact text search across all PDFs"""
        results = []
        
        for pdf_file in self.pdf_files:
            try:
                doc = fitz.open(str(pdf_file))
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    
                    # Case-insensitive search
                    if query.lower() in text.lower():
                        # Find all lines containing the query
                        lines = text.split('\n')
                        for line_num, line in enumerate(lines):
                            if query.lower() in line.lower():
                                results.append({
                                    'document': pdf_file.name,
                                    'page': page_num + 1,
                                    'line': line_num + 1,
                                    'text': line.strip(),
                                    'search_type': 'exact_match',
                                    'query': query
                                })
                
                doc.close()
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
        
        return results
    
    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings"""
        try:
            results = self.rag_system.search_documents(query, n_results)
            return results['results']
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def hybrid_search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Combine exact and semantic search results"""
        
        # Perform both searches
        exact_results = self.exact_text_search(query)
        semantic_results = self.semantic_search(query, n_results)
        
        # Combine and deduplicate results
        combined_results = {
            'query': query,
            'exact_matches': exact_results,
            'semantic_matches': semantic_results,
            'total_exact': len(exact_results),
            'total_semantic': len(semantic_results)
        }
        
        return combined_results
    
    def get_llm_response(self, query: str, search_results: Dict[str, Any]) -> str:
        """Generate LLM response based on hybrid search results"""
        
        # Prepare context from both exact and semantic matches
        context_chunks = []
        
        # Add exact matches first (higher priority)
        for result in search_results['exact_matches'][:3]:
            context_chunks.append({
                'text': result['text'],
                'document': result['document'],
                'page': result['page']
            })
        
        # Add semantic matches
        for result in search_results['semantic_matches'][:3]:
            # Avoid duplicates
            if not any(r['document'] == result['document'] and r['page'] == result['page'] 
                      for r in context_chunks):
                context_chunks.append(result)
        
        if not context_chunks:
            return "No relevant information found."
        
        # Get LLM response
        try:
            response = self.rag_system.generate_llm_response(query, context_chunks)
            return response
        except Exception as e:
            return f"Error generating response: {e}"

def test_hybrid_search():
    """Test the hybrid search system"""
    print("🔍 Hybrid Search Test")
    print("="*60)
    
    hybrid = HybridSearchRAG()
    
    # Initialize database if needed
    stats = hybrid.rag_system.get_database_stats()
    if stats.get('total_chunks', 0) == 0:
        print("📚 Processing PDFs...")
        hybrid.rag_system.process_pdfs(".")
    
    # Test query
    query = "ethanolamine"
    print(f"\n🔍 Searching for: '{query}'")
    
    results = hybrid.hybrid_search(query)
    
    print(f"\n📋 EXACT MATCHES ({results['total_exact']}):")
    for i, result in enumerate(results['exact_matches'], 1):
        print(f"  {i}. 📄 {result['document']} (Page {result['page']})")
        print(f"     Text: {result['text'][:100]}...")
    
    print(f"\n🧮 SEMANTIC MATCHES ({results['total_semantic']}):")
    for i, result in enumerate(results['semantic_matches'], 1):
        print(f"  {i}. 📄 {result['document']} (Page {result['page']})")
        print(f"     Text: {result['text'][:100]}...")
    
    # Test LLM response
    print(f"\n🤖 AI RESPONSE:")
    llm_response = hybrid.get_llm_response(query, results)
    print(llm_response)

if __name__ == "__main__":
    test_hybrid_search()