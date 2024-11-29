import logging
from typing import Dict, List, Optional
import numpy as np

class RAGRetriever:
    """
    Handles retrieval of relevant contexts for question generation using RAG.
    Works with the DocumentStore to provide context-aware information retrieval.
    """
    
    def __init__(self, document_store, retriever_config: Dict):
        """
        Initialize the RAG retriever.
        
        Args:
            document_store: Document store instance
            retriever_config: Configuration for retrieval
        """
        self.logger = logging.getLogger(__name__)
        self.document_store = document_store
        self.search_k = retriever_config['search_k']
        self.min_relevance_score = retriever_config.get('min_relevance_score', 0.0)
        
        # Statistics tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'successful_retrievals': 0,
            'average_relevance_score': 0.0
        }

    def get_relevant_contexts(
        self,
        topic: str,
        additional_context: Optional[str] = None,
        max_contexts: int = 3
    ) -> List[Dict]:
        """
        Retrieve relevant contexts for a given topic.
        
        Args:
            topic: Topic to find context for
            additional_context: Optional additional context
            max_contexts: Maximum number of contexts to return
            
        Returns:
            List of relevant contexts with metadata
        """
        try:
            self.logger.info(f"Retrieving contexts for topic: {topic}")
            self.retrieval_stats['total_queries'] += 1
            
            # Combine topic with additional context
            search_text = topic
            if additional_context:
                search_text = f"{topic} {additional_context}"
            
            # Get embeddings for search
            embedding = self._get_embedding(search_text)
            
            # Search document store
            results = self.document_store.search(
                query_embedding=embedding,
                top_k=self.search_k,
                score_threshold=self.min_relevance_score
            )
            
            # Process and filter results
            contexts = self._process_search_results(results, max_contexts)
            
            # Update statistics
            if contexts:
                self.retrieval_stats['successful_retrievals'] += 1
                avg_score = sum(ctx['relevance_score'] for ctx in contexts) / len(contexts)
                self.retrieval_stats['average_relevance_score'] = (
                    (self.retrieval_stats['average_relevance_score'] * 
                     (self.retrieval_stats['successful_retrievals'] - 1) + avg_score) /
                    self.retrieval_stats['successful_retrievals']
                )
            
            return contexts
            
        except Exception as e:
            self.logger.error(f"Error retrieving contexts: {str(e)}")
            raise

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for search text using document store's embedding manager.
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding
        """
        try:
            return self.document_store.embedding_manager.generate_embeddings(text)
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise

    def _process_search_results(
        self,
        results: List[Dict],
        max_contexts: int
    ) -> List[Dict]:
        """
        Process and filter search results.
        
        Args:
            results: Search results from document store
            max_contexts: Maximum number of contexts to return
            
        Returns:
            Processed and filtered contexts
        """
        if not results:
            return []
        
        processed_contexts = []
        seen_content = set()
        
        for result in results:
            # Stop if we have enough contexts
            if len(processed_contexts) >= max_contexts:
                break
            
            # Clean and process context
            context_text = self._clean_context(result['document'])
            
            # Skip duplicates
            if self._is_duplicate_content(context_text, seen_content):
                continue
            
            # Create context with all necessary fields
            processed_contexts.append({
                'context': context_text,
                'page_number': result['page_number'],
                'relevance_score': result['score'],
                'document_id': result.get('document_id', -1)
            })
            
            seen_content.add(context_text)
        
        return processed_contexts

    def _clean_context(self, context: str) -> str:
        """
        Clean and normalize context text.
        
        Args:
            context: Context text
            
        Returns:
            Cleaned context
        """
        if not context:
            return ""
        
        # Remove excessive whitespace
        context = " ".join(context.split())
        
        # Remove control characters
        context = "".join(char for char in context if char.isprintable() or char == '\n')
        
        return context.strip()

    def _is_duplicate_content(self, context: str, seen_content: set) -> bool:
        """
        Check if context is too similar to already seen content.
        
        Args:
            context: Context to check
            seen_content: Set of previously seen contexts
            
        Returns:
            True if context is a duplicate
        """
        # Simple Jaccard similarity check
        words1 = set(context.lower().split())
        
        for seen in seen_content:
            words2 = set(seen.lower().split())
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union > 0 and intersection / union > 0.8:
                return
