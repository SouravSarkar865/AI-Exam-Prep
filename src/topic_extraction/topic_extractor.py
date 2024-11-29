import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

class TopicExtractor:
    """
    Handles extraction of main topics from text using Ollama's language models.
    Includes enhanced error handling and connection testing capabilities.
    """
    
    def __init__(self, llm_config: Dict):
        """
        Initialize the topic extractor with configuration settings.
        
        Args:
            llm_config (Dict): Configuration dictionary containing model settings
                             (model_name, temperature, max_tokens)
        """
        # Set up logging for this class
        self.logger = logging.getLogger(__name__)
        
        # Store configuration parameters
        self.model = llm_config['model_name']
        self.temperature = llm_config['temperature']
        self.max_tokens = llm_config['max_tokens']
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Initialize connection test on startup
        if not self.test_ollama_connection():
            self.logger.warning("Initial Ollama connection test failed")

    def test_ollama_connection(self) -> bool:
        """
        Test the connection to Ollama before starting main processing.
        Sends a minimal request to verify the service is responsive.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Create a minimal test payload
            test_payload = {
                "model": self.model,
                "prompt": "test connection",
                "stream": False
            }
            
            self.logger.info("Testing Ollama connection...")
            response = requests.post(
                self.ollama_url,
                json=test_payload,
                timeout=10  # Short timeout for test
            )
            response.raise_for_status()
            
            self.logger.info("Successfully connected to Ollama")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {str(e)}")
            return False

    def extract_topics(self, text: str) -> List[str]:
        """
        Extract main topics from the provided text using Ollama.
        
        Args:
            text (str): Input text to extract topics from
            
        Returns:
            List[str]: List of extracted topics
            
        Raises:
            ConnectionError: If Ollama server is not accessible
            ValueError: If unable to extract topics
        """
        self.logger.info("Starting topic extraction...")
        
        # Verify connection before processing
        if not self.test_ollama_connection():
            raise ConnectionError("Cannot proceed with topic extraction - Ollama connection failed")
        
        try:
            # Create the prompt for topic extraction
            prompt = self._create_topic_extraction_prompt(text)
            
            # Get response from Ollama
            response = self._get_ollama_response(prompt)
            
            # Parse and validate topics
            topics = self._parse_topics(response)
            
            if not topics:
                raise ValueError("No topics could be extracted from the text")
            
            self.logger.info(f"Successfully extracted {len(topics)} topics")
            return topics
            
        except requests.exceptions.ConnectionError:
            self.logger.error("Could not connect to Ollama server")
            raise ConnectionError("Ollama server is not accessible. Please ensure it's running.")
        except Exception as e:
            self.logger.error(f"Error during topic extraction: {str(e)}")
            raise

    def _get_ollama_response(self, prompt: str) -> str:
        """
        Get response from Ollama API with enhanced error handling and logging.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Model response
            
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            # Log the attempt to connect
            self.logger.info(f"Attempting to connect to Ollama at {self.ollama_url}")
            
            # Log payload size for debugging
            prompt_size = len(prompt)
            self.logger.info(f"Sending prompt of size: {prompt_size} characters")
            
            # Set a timeout for the request
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60  # 60 seconds timeout
            )
            
            # Log the response status
            self.logger.info(f"Received response with status code: {response.status_code}")
            
            # If we get an error response, log the error details
            if response.status_code != 200:
                self.logger.error(f"Error response from Ollama: {response.text}")
                response.raise_for_status()
                
            response_data = response.json()
            
            # Verify we got the expected response format
            if 'response' not in response_data:
                raise ValueError(f"Unexpected response format: {response_data}")
                
            return response_data['response']
            
        except requests.exceptions.Timeout:
            self.logger.error("Request to Ollama timed out after 60 seconds")
            raise RuntimeError("Ollama request timed out. The service might be overwhelmed.")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error calling Ollama API: {str(e)}")
            self.logger.error(f"Full error details: {repr(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in Ollama communication: {str(e)}")
            raise

    def _create_topic_extraction_prompt(self, text: str) -> str:
        """
        Create a prompt for topic extraction with clear instructions.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Formatted prompt
        """
        return f"""As an expert in project management and content analysis, extract the main topics from the following text.
        Focus on key project management concepts and methodologies.
        
        Guidelines:
        1. Extract 5-10 main topics
        2. Each topic should be clear and specific
        3. Focus on project management concepts
        4. Return only a JSON array of topics
        5. Avoid overlapping or redundant topics
        
        Text to analyze:
        {text[:3000]}...
        
        Return the topics in this format exactly:
        ["Topic 1", "Topic 2", "Topic 3"]
        """

    def _parse_topics(self, response: str) -> List[str]:
        """
        Parse and validate topics from model response.
        
        Args:
            response (str): Model response
            
        Returns:
            List[str]: List of validated topics
            
        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Find JSON array in response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found in response")
                
            json_str = response[start_idx:end_idx]
            topics = json.loads(json_str)
            
            # Validate and clean topics
            validated_topics = []
            for topic in topics:
                if isinstance(topic, str) and topic.strip():
                    cleaned_topic = self._clean_topic(topic)
                    if cleaned_topic:
                        validated_topics.append(cleaned_topic)
            
            return validated_topics
            
        except json.JSONDecodeError:
            self.logger.error("Failed to parse JSON response from model")
            raise ValueError("Invalid response format from model")

    def _clean_topic(self, topic: str) -> str:
        """
        Clean and normalize a topic string.
        
        Args:
            topic (str): Raw topic string
            
        Returns:
            str: Cleaned topic string
        """
        # Remove leading/trailing whitespace and quotes
        topic = topic.strip().strip('"\'')
        
        # Capitalize first letter of each word
        topic = ' '.join(word.capitalize() for word in topic.split())
        
        # Remove any numbered prefixes
        topic = ' '.join(word for word in topic.split() if not word[0].isdigit())
        
        return topic

    def get_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.
        
        Returns:
            str: Current timestamp
        """
        return datetime.utcnow().isoformat() + "Z"
