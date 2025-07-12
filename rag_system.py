#!/usr/bin/env python3
"""
RAG System for PDF Document Query
Processes PDF files and creates a vector database for semantic search
"""

import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from pathlib import Path

class PDFRAGSystem:
    def __init__(self, collection_name: str = "pdf_documents"):
        """Initialize the RAG system with ChromaDB and sentence transformer"""
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Try to get existing collection or create new one
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")
    
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
    
    def process_pdfs(self, pdf_directory: str = "."):
        """Process all PDF files in the directory"""
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_directory}")
            return
        
        all_documents = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
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
            print(f"Added {len(all_documents)} text chunks to the vector database")
        else:
            print("No text chunks were extracted from PDFs")
    
    def query(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the RAG system with a question"""
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
    
    def answer_question(self, question: str) -> str:
        """Provide a formatted answer to a question"""
        results = self.query(question)
        
        if not results['results']:
            return f"No relevant information found for: {question}"
        
        # Group results by document and page
        doc_pages = {}
        for result in results['results']:
            doc = result['document']
            page = result['page']
            key = f"{doc} (page {page})"
            
            if key not in doc_pages:
                doc_pages[key] = []
            doc_pages[key].append(result['text'])
        
        # Format answer
        answer = f"Query: {question}\n\n"
        answer += "Relevant information found in:\n\n"
        
        for doc_page, texts in doc_pages.items():
            answer += f"📄 {doc_page}:\n"
            for text in texts:
                answer += f"  • {text[:200]}{'...' if len(text) > 200 else ''}\n"
            answer += "\n"
        
        return answer

def main():
    """Main function to run the RAG system"""
    print("🚀 Initializing PDF RAG System...")
    
    # Initialize the RAG system
    rag = PDFRAGSystem()
    
    # Process PDFs in current directory
    print("\n📚 Processing PDF files...")
    rag.process_pdfs(".")
    
    # Test query about ethanolamine
    print("\n🔍 Testing query...")
    question = "at what page ethanolamine was mentioned and in what document"
    
    answer = rag.answer_question(question)
    print("\n" + "="*60)
    print("QUERY RESULT:")
    print("="*60)
    print(answer)
    
    # Additional interactive mode
    print("\n" + "="*60)
    print("Interactive Query Mode (type 'quit' to exit):")
    print("="*60)
    
    while True:
        user_question = input("\n❓ Enter your question: ").strip()
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_question:
            answer = rag.answer_question(user_question)
            print("\n📋 Answer:")
            print("-" * 40)
            print(answer)

if __name__ == "__main__":
    main()