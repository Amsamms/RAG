#!/usr/bin/env python3
"""
Enhanced RAG System with LLM Integration
Supports both direct search and LLM-powered responses
"""

import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EnhancedPDFRAGSystem:
    def __init__(self, collection_name: str = "pdf_documents"):
        """Initialize the enhanced RAG system with ChromaDB, sentence transformer, and OpenAI"""
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # OpenAI configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.llm_available = True
        else:
            self.llm_available = False
            print("⚠️ OpenAI API key not found. LLM features will be disabled.")
        
        # Try to get existing collection or create new one
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✅ Using existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"🆕 Created new collection: {collection_name}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with page tracking"""
        doc = fitz.open(pdf_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Split text into chunks (paragraphs or sentences)
            chunks = self._split_text(text)
            
            for chunk_id, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    pages_data.append({
                        'text': chunk.strip(),
                        'document': os.path.basename(pdf_path),
                        'page': page_num + 1,  # 1-indexed pages
                        'chunk_id': chunk_id,
                        'file_path': pdf_path
                    })
        
        doc.close()
        return pages_data
    
    def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into smaller chunks for better vector representation"""
        # Split by paragraphs first, then by sentences if needed
        paragraphs = text.split('\n\n')
        chunks = []
        
        for paragraph in paragraphs:
            if len(paragraph) <= chunk_size:
                chunks.append(paragraph)
            else:
                # Split long paragraphs by sentences
                sentences = paragraph.split('. ')
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) <= chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_pdfs(self, pdf_directory: str = ".") -> int:
        """Process all PDF files in the directory and return count of processed chunks"""
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        
        if not pdf_files:
            print(f"📂 No PDF files found in {pdf_directory}")
            return 0
        
        all_documents = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []
        
        for pdf_file in pdf_files:
            print(f"📄 Processing: {pdf_file.name}")
            pages_data = self.extract_text_from_pdf(str(pdf_file))
            
            for i, page_data in enumerate(pages_data):
                # Create embeddings
                embedding = self.embedding_model.encode(page_data['text'])
                
                # Prepare data for ChromaDB
                doc_id = f"{page_data['document']}_page_{page_data['page']}_chunk_{page_data['chunk_id']}"
                
                all_documents.append(page_data['text'])
                all_embeddings.append(embedding.tolist())
                all_metadatas.append({
                    'document': page_data['document'],
                    'page': page_data['page'],
                    'chunk_id': page_data['chunk_id'],
                    'file_path': page_data['file_path']
                })
                all_ids.append(doc_id)
        
        # Add to ChromaDB collection
        if all_documents:
            self.collection.add(
                documents=all_documents,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print(f"✅ Added {len(all_documents)} text chunks to the vector database")
            return len(all_documents)
        else:
            print("❌ No text chunks were extracted from PDFs")
            return 0
    
    def search_documents(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """Search documents using semantic similarity"""
        # Create embedding for the question
        question_embedding = self.embedding_model.encode(question)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=n_results
        )
        
        # Format results
        response = {
            'question': question,
            'results': []
        }
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                result_item = {
                    'text': results['documents'][0][i],
                    'document': results['metadatas'][0][i]['document'],
                    'page': results['metadatas'][0][i]['page'],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                response['results'].append(result_item)
        
        return response
    
    def generate_llm_response(self, question: str, context_results: List[Dict]) -> str:
        """Generate a natural language response using OpenAI"""
        if not self.llm_available:
            return "❌ LLM service not available. Please set OPENAI_API_KEY in your .env file."
        
        # Prepare context from search results
        context_text = ""
        for i, result in enumerate(context_results, 1):
            context_text += f"\n\n[Document: {result['document']}, Page: {result['page']}]\n"
            context_text += result['text']
        
        # Create the prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided PDF document content. 
        
Use the following context to answer the question. If the answer is not in the context, say so.
Always cite the document name and page number when referencing information.

Question: {question}

Context from PDF documents:{context_text}

Answer:"""
        
        try:
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on PDF document content. Always cite sources with document name and page number."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"❌ Error generating LLM response: {str(e)}"
    
    def ask_question(self, question: str, use_llm: bool = True, n_results: int = 5) -> Dict[str, Any]:
        """Ask a question and get both search results and LLM response"""
        # Get search results
        search_results = self.search_documents(question, n_results)
        
        response = {
            'question': question,
            'search_results': search_results['results'],
            'llm_response': None,
            'llm_available': self.llm_available
        }
        
        # Generate LLM response if requested and available
        if use_llm and self.llm_available and search_results['results']:
            response['llm_response'] = self.generate_llm_response(question, search_results['results'])
        elif use_llm and not self.llm_available:
            response['llm_response'] = "❌ LLM service not available. Please configure OpenAI API key."
        elif use_llm and not search_results['results']:
            response['llm_response'] = "❌ No relevant documents found to answer the question."
        
        return response
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the document database"""
        try:
            collection_count = self.collection.count()
            return {
                'total_chunks': collection_count,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model_name,
                'llm_available': self.llm_available,
                'llm_model': self.openai_model if self.llm_available else None
            }
        except Exception as e:
            return {'error': str(e)}

def main():
    """Command line interface for the enhanced RAG system"""
    print("🚀 Enhanced PDF RAG System with LLM Integration")
    print("="*60)
    
    # Initialize the RAG system
    rag = EnhancedPDFRAGSystem()
    
    # Show database stats
    stats = rag.get_database_stats()
    if 'error' not in stats:
        print(f"📊 Database Stats:")
        print(f"   • Total chunks: {stats['total_chunks']}")
        print(f"   • Embedding model: {stats['embedding_model']}")
        print(f"   • LLM available: {stats['llm_available']}")
        if stats['llm_available']:
            print(f"   • LLM model: {stats['llm_model']}")
    
    # Process PDFs if database is empty
    if stats.get('total_chunks', 0) == 0:
        print("\n📚 Processing PDF files...")
        chunks_added = rag.process_pdfs(".")
        print(f"✅ Processed {chunks_added} chunks")
    
    # Interactive query loop
    print("\n" + "="*60)
    print("🔍 Interactive Query Mode")
    print("Commands:")
    print("  • Type your question")
    print("  • 'stats' - Show database statistics")
    print("  • 'quit' or 'exit' - Exit the program")
    print("="*60)
    
    while True:
        user_question = input("\n❓ Enter your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        elif user_question.lower() == 'stats':
            stats = rag.get_database_stats()
            print(f"📊 Database Statistics: {stats}")
            continue
        elif not user_question:
            continue
        
        print("\n🔍 Searching...")
        result = rag.ask_question(user_question, use_llm=True)
        
        print("\n" + "="*60)
        print("📋 SEARCH RESULTS:")
        print("="*60)
        
        if result['search_results']:
            for i, res in enumerate(result['search_results'], 1):
                print(f"{i}. 📄 {res['document']} (Page {res['page']})")
                print(f"   {res['text'][:200]}...")
                print()
        else:
            print("❌ No relevant documents found.")
        
        if result['llm_response']:
            print("🤖 AI RESPONSE:")
            print("="*60)
            print(result['llm_response'])
        
        print("="*60)

if __name__ == "__main__":
    main()