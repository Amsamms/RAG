#!/usr/bin/env python3
"""
FAISS Vector Store - ChromaDB Replacement for Streamlit Cloud Compatibility
Provides the same interface as ChromaDB but uses FAISS for vector storage
"""

import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
from sentence_transformers import SentenceTransformer


class FaissVectorStore:
    """FAISS-based vector store that mimics ChromaDB interface"""
    
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "./faiss_db"
        self.embedding_model = None
        self.index = None
        self.documents = []  # Store document metadata
        self.embeddings = []  # Store embeddings
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(exist_ok=True)
        
        # Initialize FAISS index
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize FAISS index"""
        # Create a flat L2 index for similarity search
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Try to load existing data
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Load existing index and metadata from disk"""
        index_path = Path(self.persist_directory) / f"{self.collection_name}.index"
        metadata_path = Path(self.persist_directory) / f"{self.collection_name}_metadata.json"
        embeddings_path = Path(self.persist_directory) / f"{self.collection_name}_embeddings.pkl"
        
        if index_path.exists() and metadata_path.exists() and embeddings_path.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    self.documents = json.load(f)
                
                # Load embeddings
                with open(embeddings_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
                    
                print(f"✅ Loaded existing collection: {self.collection_name} with {len(self.documents)} documents")
            except Exception as e:
                print(f"⚠️ Could not load existing data: {e}")
                self._reset_collection()
        else:
            print(f"🆕 Created new collection: {self.collection_name}")
    
    def _save_to_disk(self):
        """Save index and metadata to disk"""
        try:
            index_path = Path(self.persist_directory) / f"{self.collection_name}.index"
            metadata_path = Path(self.persist_directory) / f"{self.collection_name}_metadata.json"
            embeddings_path = Path(self.persist_directory) / f"{self.collection_name}_embeddings.pkl"
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(self.documents, f, indent=2)
            
            # Save embeddings
            with open(embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
                
        except Exception as e:
            print(f"⚠️ Could not save to disk: {e}")
    
    def _reset_collection(self):
        """Reset the collection to empty state"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.embeddings = []
    
    def add(self, documents: List[str], metadatas: List[Dict[str, Any]], 
            ids: List[str], embeddings: Optional[List[List[float]]] = None):
        """Add documents to the vector store"""
        
        if embeddings is None:
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = self.embedding_model.encode(documents).tolist()
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store metadata and embeddings
        for i, (doc, metadata, doc_id, embedding) in enumerate(zip(documents, metadatas, ids, embeddings)):
            document_data = {
                'id': doc_id,
                'document': doc,
                'metadata': metadata,
                'index_position': len(self.documents)  # Track position in FAISS index
            }
            self.documents.append(document_data)
            self.embeddings.append(embedding)
        
        # Save to disk
        self._save_to_disk()
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 10,
              where: Optional[Dict[str, Any]] = None) -> Dict[str, List]:
        """Query the vector store for similar documents"""
        
        if len(self.documents) == 0:
            return {
                'documents': [[]],
                'metadatas': [[]],
                'ids': [[]],
                'distances': [[]]
            }
        
        # Convert query embeddings to numpy array
        query_array = np.array(query_embeddings, dtype=np.float32)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_array, min(n_results, len(self.documents)))
        
        # Prepare results
        documents = []
        metadatas = []
        ids = []
        result_distances = []
        
        for query_idx in range(len(query_embeddings)):
            query_docs = []
            query_metas = []
            query_ids = []
            query_dists = []
            
            for i, idx in enumerate(indices[query_idx]):
                if idx >= 0 and idx < len(self.documents):  # Valid index
                    doc_data = self.documents[idx]
                    
                    # Apply metadata filtering if specified
                    if where is None or self._matches_filter(doc_data['metadata'], where):
                        query_docs.append(doc_data['document'])
                        query_metas.append(doc_data['metadata'])
                        query_ids.append(doc_data['id'])
                        query_dists.append(float(distances[query_idx][i]))
            
            documents.append(query_docs)
            metadatas.append(query_metas)
            ids.append(query_ids)
            result_distances.append(query_dists)
        
        return {
            'documents': documents,
            'metadatas': metadatas,
            'ids': ids,
            'distances': result_distances
        }
    
    def _matches_filter(self, metadata: Dict[str, Any], where: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter criteria"""
        for key, value in where.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def count(self) -> int:
        """Return the number of documents in the collection"""
        return len(self.documents)
    
    def delete_collection(self):
        """Delete the entire collection"""
        # Remove files from disk
        try:
            index_path = Path(self.persist_directory) / f"{self.collection_name}.index"
            metadata_path = Path(self.persist_directory) / f"{self.collection_name}_metadata.json"
            embeddings_path = Path(self.persist_directory) / f"{self.collection_name}_embeddings.pkl"
            
            for path in [index_path, metadata_path, embeddings_path]:
                if path.exists():
                    path.unlink()
        except Exception as e:
            print(f"⚠️ Error deleting collection files: {e}")
        
        # Reset in-memory data
        self._reset_collection()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            'collection_name': self.collection_name,
            'total_documents': self.count(),
            'index_dimension': self.dimension,
            'persist_directory': self.persist_directory
        }


class FaissClient:
    """FAISS client that mimics ChromaDB client interface"""
    
    def __init__(self, persist_directory: str = "./faiss_db"):
        self.persist_directory = persist_directory
        self.collections = {}
    
    def create_collection(self, name: str) -> FaissVectorStore:
        """Create a new collection"""
        collection = FaissVectorStore(collection_name=name, persist_directory=self.persist_directory)
        self.collections[name] = collection
        return collection
    
    def get_collection(self, name: str) -> FaissVectorStore:
        """Get an existing collection"""
        if name in self.collections:
            return self.collections[name]
        
        # Try to load from disk
        collection = FaissVectorStore(collection_name=name, persist_directory=self.persist_directory)
        self.collections[name] = collection
        return collection
    
    def get_or_create_collection(self, name: str) -> FaissVectorStore:
        """Get collection if it exists, otherwise create it"""
        try:
            return self.get_collection(name)
        except Exception:
            return self.create_collection(name)
    
    def delete_collection(self, name: str):
        """Delete a collection"""
        if name in self.collections:
            self.collections[name].delete_collection()
            del self.collections[name]