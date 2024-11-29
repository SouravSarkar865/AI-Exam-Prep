import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from pdf_processor.extractor import PDFExtractor
from topic_extraction.topic_extractor import TopicExtractor
from embeddings.embedding_manager import EmbeddingManager
from rag.document_store import DocumentStore
from rag.retriever import RAGRetriever
from question_generation.generator import QuestionGenerator
from utils.helpers import setup_logging, save_json

class QuestionGenerationPipeline:
    """
    Main pipeline for generating questions from project management documents.
    Coordinates the entire process from PDF processing to question generation,
    maintaining document structure and page number tracking throughout.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the pipeline with configuration and setup components.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            # Load configuration and setup logging
            self.config = self._load_config(config_path)
            setup_logging(self.config['logging'])
            self.logger = logging.getLogger(__name__)
            
            # Initialize pipeline components
            self.logger.info("Initializing pipeline components...")
            
            # Create each component in the proper order
            self.pdf_extractor = PDFExtractor()
            self.topic_extractor = TopicExtractor(self.config['llm'])
            self.embedding_manager = EmbeddingManager(self.config['embeddings'])
            
            # Initialize document store with embedding manager
            self.document_store = DocumentStore(
                self.config['vector_store'],
                embedding_manager=self.embedding_manager
            )
            
            # Initialize RAG retriever
            self.rag_retriever = RAGRetriever(
                self.document_store,
                self.config['rag']['retriever']
            )
            
            # Initialize question generator
            self.question_generator = QuestionGenerator(
                self.config['llm'],
                self.config['question_generation']
            )
            
            self.logger.info("Pipeline initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise RuntimeError(f"Pipeline initialization failed: {str(e)}")

    def _load_config(self, config_path: str) -> Dict:
        """
        Load and validate configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validated configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validate essential configuration sections
            required_sections = ['llm', 'embeddings', 'vector_store', 'pdf', 'rag']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required configuration section: {section}")
                    
            return config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {str(e)}")

    def _create_chunks(self, documents: List[Dict[str, Union[str, int]]]) -> List[Dict]:
        """
        Split documents into chunks while preserving page number information.
        
        Args:
            documents: List of dictionaries containing text and page numbers
            
        Returns:
            List of dictionaries with chunked text and page numbers
        """
        self.logger.info("Creating text chunks...")
        
        chunked_documents = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['pdf']['chunk_size'],
            chunk_overlap=self.config['pdf']['chunk_overlap']
        )
        
        for doc in tqdm(documents, desc="Processing document chunks"):
            # Split the text into chunks
            text_chunks = splitter.split_text(doc['text'])
            
            # Create new document dictionaries for each chunk
            for chunk in text_chunks:
                chunked_documents.append({
                    'text': chunk,
                    'page_number': doc['page_number']
                })
        
        self.logger.info(f"Created {len(chunked_documents)} text chunks")
        return chunked_documents

    def generate_topics(self, text: str) -> Dict:
        """
        Extract main topics from the text.
        
        Args:
            text: Input text to extract topics from
            
        Returns:
            Dictionary containing extracted topics and metadata
        """
        self.logger.info("Extracting main topics from text...")
        
        try:
            # Extract topics using the topic extractor
            topics = self.topic_extractor.extract_topics(text)
            
            # Prepare topics data structure
            topics_data = {
                "book_title": "Project Management Professional Guide",
                "total_topics": len(topics),
                "extraction_timestamp": self.topic_extractor.get_timestamp(),
                "main_topics": topics
            }
            
            # Save topics to file
            save_json(topics_data, self.config['output']['topics_path'])
            self.logger.info(f"Successfully extracted {len(topics)} topics")
            return topics_data
            
        except Exception as e:
            self.logger.error(f"Topic generation failed: {str(e)}")
            raise

    def setup_rag_pipeline(self, chunks: List[Dict]):
        """
        Set up the RAG pipeline with document chunks.
        
        Args:
            chunks: List of text chunks with page numbers
        """
        self.logger.info("Setting up RAG pipeline...")
        try:
            # Extract text from chunks for embedding generation
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings for all chunks
            embeddings = self.embedding_manager.generate_embeddings(texts)
            
            # Add documents to document store
            self.document_store.add_documents(chunks, embeddings)
            
            self.logger.info("RAG pipeline setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"RAG pipeline setup failed: {str(e)}")
            raise RuntimeError(f"Failed to setup RAG pipeline: {str(e)}")

    def generate_questions(self, topics: List[str]) -> Dict:
        """
        Generate questions for each topic using RAG.
        
        Args:
            topics: List of topics to generate questions for
            
        Returns:
            Dictionary containing generated questions and metadata
        """
        self.logger.info("Starting question generation process...")
        
        questions_data = {
            "metadata": {
                "generated_at": self.question_generator.get_timestamp(),
                "total_questions": 0,
                "book_title": "Project Management Professional Guide",
                "generation_method": "RAG Pipeline",
                "embedding_model": self.config['embeddings']['model_name'],
                "vector_store": "FAISS"
            },
            "questions": []
        }

        try:
            for topic in tqdm(topics, desc="Generating questions by topic"):
                # Get relevant context for each topic
                contexts = self.rag_retriever.get_relevant_contexts(topic)
                
                # Generate questions using context
                topic_questions = self.question_generator.generate_questions(
                    topic,
                    contexts,
                    self.config['question_generation']['questions_per_topic']
                )
                
                questions_data["questions"].extend(topic_questions)

            # Update metadata and save results
            questions_data["metadata"]["total_questions"] = len(questions_data["questions"])
            save_json(questions_data, self.config['output']['questions_path'])
            
            self.logger.info(f"Successfully generated {len(questions_data['questions'])} questions")
            return questions_data
            
        except Exception as e:
            self.logger.error(f"Question generation failed: {str(e)}")
            raise

    def run(self) -> Tuple[Dict, Dict]:
        """
        Execute the complete pipeline.
        
        Returns:
            Tuple containing topics data and questions data
        """
        try:
            # Extract text from PDF
            self.logger.info("Starting PDF extraction...")
            documents = self.pdf_extractor.extract_text(self.config['pdf']['input_path'])
            
            # Create chunks while preserving page numbers
            chunks = self._create_chunks(documents)
            
            # Combine all text for topic extraction
            full_text = ' '.join(doc['text'] for doc in documents)
            
            # Generate topics
            topics_data = self.generate_topics(full_text)
            
            # Setup RAG pipeline
            self.setup_rag_pipeline(chunks)
            
            # Generate questions
            questions_data = self.generate_questions(topics_data["main_topics"])
            
            self.logger.info("Pipeline completed successfully!")
            return topics_data, questions_data
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise

def main():
    """
    Main entry point for the question generation system.
    Sets up the environment and runs the pipeline with error handling.
    """
    try:
        # Create necessary directories
        os.makedirs("results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data/vector_store", exist_ok=True)
        
        # Initialize pipeline
        pipeline = QuestionGenerationPipeline()
        
        # Verify Ollama connection
        if not pipeline.topic_extractor.test_ollama_connection():
            print("\nFailed to connect to Ollama. Please ensure:")
            print("1. Ollama is installed and running")
            print("2. The service is accessible at localhost:11434")
            print("3. The required model (mistral) is installed")
            exit(1)
        
        # Execute pipeline
        topics_data, questions_data = pipeline.run()
        
        # Report results
        print("\nPipeline completed successfully!")
        print(f"Generated {len(topics_data['main_topics'])} topics")
        print(f"Generated {questions_data['metadata']['total_questions']} questions")
        print("\nResults have been saved to:")
        print(f"- Topics: {pipeline.config['output']['topics_path']}")
        print(f"- Questions: {pipeline.config['output']['questions_path']}")
        
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
