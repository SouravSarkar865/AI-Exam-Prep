import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import requests
from tqdm import tqdm

class QuestionGenerator:
    """
    Generates questions using Ollama and RAG contexts.
    This class ensures consistent handling of context data structures
    and maintains proper metadata throughout the generation process.
    """
    
    def __init__(self, llm_config: Dict, generator_config: Dict):
        """
        Initialize the question generator with LLM and generation settings.
        
        Args:
            llm_config: Configuration for the language model
            generator_config: Configuration for question generation
        """
        self.logger = logging.getLogger(__name__)
        
        # Store configuration parameters
        self.model = llm_config['model_name']
        self.temperature = llm_config['temperature']
        self.max_tokens = llm_config['max_tokens']
        self.questions_per_topic = generator_config['questions_per_topic']
        self.mcq_options = generator_config['mcq_options']
        
        # Ollama API endpoint
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Question ID counter
        self.question_counter = 0

    def generate_questions(
        self,
        topic: str,
        contexts: List[Dict],
        num_questions: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate questions for a topic using provided contexts.
        
        Args:
            topic: Topic to generate questions about
            contexts: List of relevant contexts with their metadata
            num_questions: Optional number of questions to generate
            
        Returns:
            List of generated questions with metadata
        """
        if not contexts:
            self.logger.warning(f"No contexts provided for topic: {topic}")
            return []
        
        num_questions = num_questions or self.questions_per_topic
        
        try:
            questions = []
            # Sort contexts by relevance score
            sorted_contexts = sorted(
                contexts,
                key=lambda x: x['relevance_score'],
                reverse=True
            )
            
            # Generate questions in smaller batches for better quality
            batch_size = 3
            for i in range(0, num_questions, batch_size):
                batch_count = min(batch_size, num_questions - i)
                
                # Generate batch of questions
                batch_questions = self._generate_question_batch(
                    topic,
                    sorted_contexts,
                    batch_count
                )
                
                questions.extend(batch_questions)
            
            # Validate and clean questions
            valid_questions = [q for q in questions if self._validate_question(q)]
            
            return valid_questions[:num_questions]
            
        except Exception as e:
            self.logger.error(f"Error generating questions: {str(e)}")
            raise

    def _create_question_prompt(
        self,
        topic: str,
        contexts: List[Dict],
        batch_size: int
    ) -> str:
        """
        Create a detailed prompt for question generation.
        
        Args:
            topic: Topic for questions
            contexts: Context information with metadata
            batch_size: Number of questions to generate
            
        Returns:
            Formatted prompt string
        """
        # Prepare context information with page numbers
        formatted_contexts = []
        for ctx in contexts:
            # Use 'context' key instead of 'document'
            formatted_contexts.append(
                f"[Page {ctx['page_number']}] {ctx['context']}"
            )
        
        context_text = "\n\n".join(formatted_contexts)
        
        return f"""Generate {batch_size} multiple-choice questions about {topic} that test understanding of key project management concepts. While your questions should be based on the provided content, frame them to test practical understanding rather than mere recall.

Context:
{context_text}

Requirements:
1. Generate {batch_size} questions that test conceptual understanding
2. Each question should have {self.mcq_options} options
3. Include the correct answer and a detailed explanation
4. Frame questions as if testing real-world understanding
5. Avoid phrases like "according to the text" or "as stated in"
6. Focus on testing application of concepts rather than recall

Each question should test whether the learner truly understands the concept and can apply it, not just remember where they read about it.

Return the questions in this JSON format:
[{{
    "question": "Question text",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_answer": "A",
    "explanation": "Detailed explanation with page reference",
    "source": {{"page_number": 123}}
}}]"""

    def _generate_question_batch(
        self,
        topic: str,
        contexts: List[Dict],
        batch_size: int
    ) -> List[Dict]:
        """
        Generate a batch of questions using provided contexts.
        
        Args:
            topic: Topic for questions
            contexts: Relevant contexts with metadata
            batch_size: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        prompt = self._create_question_prompt(topic, contexts, batch_size)
        
        try:
            # Get response from Ollama
            response = self._get_ollama_response(prompt)
            
            # Parse questions from response
            questions = self._parse_questions(response, topic, contexts)
            
            return questions
            
        except Exception as e:
            self.logger.error(f"Error in batch generation: {str(e)}")
            return []

    def _get_ollama_response(self, prompt: str) -> str:
        """
        Get response from Ollama API with error handling.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Model response
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()['response']
            
        except requests.exceptions.Timeout:
            self.logger.error("Request to Ollama timed out")
            raise RuntimeError("Ollama request timed out")
        except Exception as e:
            self.logger.error(f"Error calling Ollama API: {str(e)}")
            raise

    def _parse_questions(
        self,
        response: str,
        topic: str,
        contexts: List[Dict]
    ) -> List[Dict]:
        """
        Parse questions from model response and add metadata.
        
        Args:
            response: Model response
            topic: Question topic
            contexts: Source contexts with metadata
            
        Returns:
            List of parsed and formatted questions
        """
        try:
            # Extract JSON array from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found in response")
                
            questions_json = response[start_idx:end_idx]
            questions = json.loads(questions_json)
            
            # Format and validate each question
            formatted_questions = []
            for q in questions:
                question_dict = self._format_question(q, topic, contexts)
                if question_dict:
                    formatted_questions.append(question_dict)
            
            return formatted_questions
            
        except json.JSONDecodeError:
            self.logger.error("Failed to parse JSON response")
            return []

    def _format_question(
        self,
        question: Dict,
        topic: str,
        contexts: List[Dict]
    ) -> Optional[Dict]:
        """
        Format and validate a single question.
        
        Args:
            question: Raw question dictionary
            topic: Question topic
            contexts: Source contexts with metadata
            
        Returns:
            Formatted question dictionary or None if invalid
        """
        try:
            required_fields = [
                'question', 'options', 'correct_answer',
                'explanation', 'source'
            ]
            if not all(field in question for field in required_fields):
                return None
            
            # Get page number from source or find most relevant context
            page_number = question['source'].get('page_number', -1)
            if page_number == -1:
                for ctx in contexts:
                    if ctx['context'].lower() in question['explanation'].lower():
                        page_number = ctx['page_number']
                        break
            
            self.question_counter += 1
            formatted = {
                'id': f"Q{self.question_counter}",
                'topic': topic,
                'type': 'MCQ',
                'question': question['question'].strip(),
                'options': [opt.strip() for opt in question['options']],
                'correct_answer': question['correct_answer'].strip(),
                'explanation': f"{question['explanation'].strip()} (Page {page_number})",
                'source': {
                    'page_number': page_number,
                    'confidence_score': 0.9
                }
            }
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting question: {str(e)}")
            return None

    def _validate_question(self, question: Dict) -> bool:
        """
        Validate a formatted question for completeness and accuracy.
        
        Args:
            question: Question to validate
            
        Returns:
            True if question is valid
        """
        try:
            # Check required fields
            required_fields = [
                'id', 'topic', 'type', 'question', 'options',
                'correct_answer', 'explanation', 'source'
            ]
            if not all(key in question for key in required_fields):
                return False
            
            # Validate options
            if len(question['options']) != self.mcq_options:
                return False
            
            # Validate correct answer
            if not any(opt.startswith(f"{question['correct_answer']})") 
                      for opt in question['options']):
                return False
            
            # Validate content length
            if len(question['question']) < 10:
                return False
            
            if len(question['explanation']) < 20:
                return False
            
            # Validate page number
            if question['source']['page_number'] < 0:
                return False
            
            return True
            
        except Exception:
            return False

    def get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"
