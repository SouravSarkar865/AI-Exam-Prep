import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

class EmbeddingManager:
    """Manages text embeddings using sentence-transformers."""
    
    def __init__(self, embedding_config: dict):
        """
        Initialize the embedding manager.
        
        Args:
            embedding_config (dict): Configuration for embeddings
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = embedding_config['model_name']
        self.max_seq_length = embedding_config['max_seq_length']
        self.batch_size = embedding_config['batch_size']
        
        self.model = self._load_model()
        
    def _load_model(self) -> SentenceTransformer:
        """
        Load the sentence transformer model.
        
        Returns:
            SentenceTransformer: Loaded model
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            model.max_seq_length = self.max_seq_length
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Could not load embedding model: {str(e)}")

    def generate_embeddings(
        self, 
        texts: Union[List[str], str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for input texts.
        
        Args:
            texts (Union[List[str], str]): Input text or list of texts
            show_progress (bool): Whether to show progress bar
            
        Returns:
            np.ndarray: Generated embeddings
            
        Raises:
            ValueError: If input texts are invalid
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a string or list of strings")
            
        try:
            self.logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Split texts into batches
            embeddings = []
            for i in tqdm(
                range(0, len(texts), self.batch_size),
                desc="Generating embeddings",
                disable=not show_progress
            ):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self._generate_batch_embeddings(batch)
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def _generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts (List[str]): Batch of texts
            
        Returns:
            List[np.ndarray]: Batch embeddings
        """
        try:
            # Normalize and preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model.encode(
                    processed_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error in batch embedding generation: {str(e)}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before generating embeddings.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not text.strip():
            return ""
            
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if needed
        if len(text) > self.max_seq_length:
            text = text[:self.max_seq_length]
        
        return text

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            int: Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()

    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            dict: Model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "max_sequence_length": self.max_seq_length,
            "model_type": type(self.model).__name__
        }

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
            
        Returns:
            float: Cosine similarity score
            
        Raises:
            ValueError: If embeddings have different dimensions
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError("Embeddings must have the same dimensions")
            
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return float(similarity)

    def batch_compute_similarity(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Compute similarities between a query and corpus embeddings.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            corpus_embeddings (np.ndarray): Corpus embeddings
            top_k (int): Number of top similar results to return
            
        Returns:
            List[tuple]: List of (index, similarity_score) tuples
        """
        # Normalize embeddings
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, np.newaxis]
        
        # Compute similarities
        similarities = np.dot(corpus_embeddings, query_embedding)
        
        # Get top k indices and scores
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        return list(zip(top_indices.tolist(), top_similarities.tolist()))
