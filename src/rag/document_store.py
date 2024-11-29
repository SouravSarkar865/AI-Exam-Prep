import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
from tqdm import tqdm

class DocumentStore:
    """
    A comprehensive document storage system using FAISS for vector similarity search.
    This class manages document storage, embedding indexing, and retrieval while
    maintaining consistent metadata throughout all operations.
    """
    
    def __init__(self, vector_store_config: Dict, embedding_manager=None):
        """
        Initialize the document store with configuration and embedding manager.
        
        Args:
            vector_store_config: Configuration settings
            embedding_manager: Optional embedding manager instance
        """
        self.logger = logging.getLogger(__name__)
        
        # Store configuration and essential parameters
        self.config = vector_store_config
        self.dimension = vector_store_config['dimension']
        self.index_path = vector_store_config['index_path']
        self.embedding_manager = embedding_manager
        
        # Initialize storage containers
        self.index = None
        self.documents = []  # Stores document texts
        self.page_numbers = []  # Stores corresponding page numbers
        self.metadata = []  # Stores additional metadata
        
        # Set batch processing parameters
        self.batch_size = 32
        
        # Initialize index
        self.initialize_index()

    def initialize_index(self):
        """Initialize or load the FAISS index for vector similarity search."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Clear existing data
            self.documents = []
            self.page_numbers = []
            self.metadata = []
            
            # Remove existing index if present
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
                self.logger.info("Removed existing index file")
            
            # Create new IVF index
            self.logger.info("Creating new FAISS index")
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                self.config['nlist'],
                faiss.METRIC_INNER_PRODUCT
            )
            
            # Train index with random vectors
            training_vectors = np.random.rand(
                max(self.config['nlist'] * 10, 1000),
                self.dimension
            ).astype(np.float32)
            
            self.logger.info("Training FAISS index...")
            self.index.train(training_vectors)
            
            # Set search parameters
            self.index.nprobe = self.config['nprobe']
            
            # Save initial index
            self._save_index()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {str(e)}")
            raise RuntimeError(f"FAISS index initialization failed: {str(e)}")

    def add_documents(
        self,
        documents: List[Dict[str, Union[str, int]]],
        embeddings: np.ndarray
    ):
        """
        Add documents and their embeddings to the store.
        
        Args:
            documents: List of documents with text and page numbers
            embeddings: Document embeddings
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        try:
            # Process documents in batches
            for i in tqdm(range(0, len(documents), self.batch_size),
                         desc="Adding documents to store"):
                batch_end = min(i + self.batch_size, len(documents))
                doc_batch = documents[i:batch_end]
                emb_batch = embeddings[i:batch_end]
                
                # Process batch
                self._add_batch(doc_batch, emb_batch)
            
            # Verify final integrity
            if not self.verify_store_integrity():
                raise RuntimeError(
                    f"Final integrity check failed. "
                    f"Index: {self.index.ntotal}, "
                    f"Docs: {len(self.documents)}, "
                    f"Pages: {len(self.page_numbers)}"
                )
            
            # Save updated index
            self._save_index()
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            raise

    def _add_batch(
        self,
        doc_batch: List[Dict[str, Union[str, int]]],
        emb_batch: np.ndarray
    ):
        """
        Add a batch of documents with their embeddings.
        
        Args:
            doc_batch: Batch of documents
            emb_batch: Corresponding embeddings
        """
        curr_docs_count = len(self.documents)
        
        try:
            # Extract text and page numbers
            texts = [doc['text'] for doc in doc_batch]
            pages = [doc['page_number'] for doc in doc_batch]
            
            # Add to storage
            self.documents.extend(texts)
            self.page_numbers.extend(pages)
            
            # Process embeddings
            normalized_emb = self._normalize_embeddings(emb_batch)
            self.index.add(normalized_emb)
            
            # Verify batch integrity
            if not self.verify_store_integrity():
                raise RuntimeError("Batch integrity check failed")
            
        except Exception as e:
            # Rollback on failure
            self.documents = self.documents[:curr_docs_count]
            self.page_numbers = self.page_numbers[:curr_docs_count]
            raise

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar documents with complete metadata.
        
        Args:
            query_embedding: Query vector
            top_k: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of results with document text, metadata, and scores
        """
        try:
            # Verify store integrity
            if not self.verify_store_integrity():
                raise RuntimeError("Document store integrity check failed")
            
            if len(self.documents) == 0:
                self.logger.warning("No documents in store")
                return []
            
            # Prepare query
            query_embedding = self._normalize_embeddings(query_embedding.reshape(1, -1))
            adjusted_top_k = min(top_k, len(self.documents))
            
            # Perform search
            scores, indices = self.index.search(query_embedding, adjusted_top_k)
            
            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= score_threshold:
                    if 0 <= idx < len(self.documents):
                        results.append({
                            'document': self.documents[idx],
                            'page_number': self.page_numbers[idx],
                            'document_id': int(idx),
                            'score': float(score)
                        })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search operation failed: {str(e)}")
            raise

    def verify_store_integrity(self) -> bool:
        """
        Verify synchronization between index and document storage.
        
        Returns:
            bool: True if store is in a consistent state
        """
        if self.index is None:
            self.logger.error("Store integrity check failed: Index is None")
            return False
            
        index_size = self.index.ntotal
        docs_size = len(self.documents)
        pages_size = len(self.page_numbers)
        
        sizes_match = (index_size == docs_size == pages_size)
        
        if not sizes_match:
            self.logger.error(
                f"Store integrity check failed: Mismatched sizes - "
                f"Index: {index_size}, Docs: {docs_size}, "
                f"Pages: {pages_size}"
            )
            return False
        
        return True

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for consistent similarity calculations."""
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    def _save_index(self):
        """Save FAISS index to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            self.logger.debug(f"Saved index to {self.index_path}")
        except Exception as e:
            self.logger.error(f"Failed to save index: {str(e)}")
            raise RuntimeError(f"Failed to save FAISS index: {str(e)}")

    def clear(self):
        """Reset the document store to a clean initial state."""
        try:
            self.documents = []
            self.page_numbers = []
            self.metadata = []
            self.initialize_index()
            self.logger.info("Document store cleared successfully")
        except Exception as e:
            self.logger.error(f"Failed to clear document store: {str(e)}")
            raise

    def get_stats(self) -> Dict:
        """Get current store statistics and state information."""
        return {
            'total_documents': len(self.documents),
            'dimension': self.dimension,
            'index_type': type(self.index).__name__,
            'index_size': os.path.getsize(self.index_path) if os.path.exists(self.index_path) else 0,
            'integrity_check': self.verify_store_integrity()
        }
