# Project Management Book Question Generator

A sophisticated RAG-based system that automatically processes project management books to generate high-quality multiple-choice questions. This system leverages local Large Language Models through Ollama, combining the power of modern AI with efficient information retrieval to create educational content.

## Overview

This project implements a two-level question generation system that processes project management texts intelligently. At its core, it uses Retrieval-Augmented Generation (RAG) to ensure questions are accurately grounded in the source material. The system first extracts key topics from the text, then generates relevant multiple-choice questions while maintaining references to specific pages and sections.

### Key Features

The system offers a comprehensive set of capabilities:

- Intelligent PDF Processing: Extracts and cleanses text while preserving structural information
- Topic Identification: Automatically identifies main topics and subtopics using AI
- Context-Aware Generation: Creates questions based on relevant context through RAG
- Quality Assurance: Implements validation checks for question quality and accuracy
- Source Tracking: Maintains references to source material with page numbers
- Detailed Explanations: Provides comprehensive explanations for each answer

## Technical Requirements

Before installation, ensure your system meets these requirements:

- Python 3.9 or higher
- Operating System: Linux, macOS, or Windows
- RAM: Minimum 8GB (16GB recommended for optimal performance)
- Storage: 15GB free space (for models and vector store)
- Optional: CUDA-compatible GPU for improved performance

## Installation Guide

Follow these steps to set up the project:

1. Clone the Repository:
```bash
git clone <repository-url>
cd project-management-qgen
```

2. Environment Setup:
```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# For Linux/macOS:
source venv/bin/activate
# For Windows:
.\venv\Scripts\activate

# Install dependencies and set up project structure
python setup.py
```

3. Install Ollama:
```bash
# Make installation script executable (Linux/macOS)
chmod +x install_ollama.sh

# Run installation script
./install_ollama.sh
```

For Windows and macOS users, if automatic installation fails:
1. Visit https://ollama.com/download
2. Download the appropriate installer (.msi for Windows, .dmg for macOS)
3. Run the installer and follow system-specific instructions
4. After installation, open a terminal and run: `ollama pull mixtral`

## System Configuration

The system's behavior can be customized through the `config/config.yaml` file:

```yaml
# LLM Configuration
llm:
  model_name: "mistral"  # Alternative: "llama2"
  temperature: 0.7      # Controls creativity (0.0-1.0)
  max_tokens: 2048      # Maximum response length
  context_window: 8192  # Context window size

# Embedding Settings
embeddings:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  max_seq_length: 384
  batch_size: 32

# Question Generation Parameters
question_generation:
  questions_per_topic: 10
  mcq_options: 4
  difficulty_levels: ["easy", "medium", "hard"]

# Vector Store Settings
vector_store:
  dimension: 768
  similarity_metric: "cosine"
```

## Using the System

To generate questions from your project management book:

1. Prepare Your Document:
```bash
# Copy your PDF to the resources directory
cp "your_project_management.pdf" resources/Project_Management.pdf
```

2. Run the Generation Pipeline:
```bash
python src/main.py
```

3. Access Generated Content:
The system creates two main output files in the `results` directory:
- `topics.json`: Contains extracted topics and their hierarchy
- `questions.json`: Contains generated questions with answers and explanations

## Output Format Examples

### Topics JSON Structure:
```json
{
    "book_title": "Project Management Professional Guide",
    "total_topics": 5,
    "extraction_timestamp": "2024-11-25T10:00:00Z",
    "main_topics": [
        "Project Initiation",
        "Project Planning",
        "Project Execution"
    ]
}
```

### Questions JSON Structure:
```json
{
    "metadata": {
        "generated_at": "2024-11-25T10:30:00Z",
        "total_questions": 50,
        "generation_method": "RAG Pipeline",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2"
    },
    "questions": [
        {
            "id": "Q1",
            "topic": "Project Initiation",
            "type": "MCQ",
            "question": "What is the primary purpose of a project charter?",
            "options": [
                "A) Define project budget",
                "B) Formally authorize the project",
                "C) Assign project team",
                "D) Schedule meetings"
            ],
            "correct_answer": "B",
            "explanation": "A project charter formally authorizes the project...",
            "source": {
                "page_number": 13,
                "confidence_score": 0.92
            }
        }
    ]
}
```

## Project Structure

```
project_root/
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── approaches.txt          # Technical approach documentation
├── install_ollama.sh       # Ollama installation script
├── setup.py                # Project setup script
├── config/
│   └── config.yaml         # Configuration settings
├── src/
│   ├── main.py            # Main entry point
│   ├── pdf_processor/     # PDF handling
│   ├── topic_extraction/  # Topic identification
│   ├── embeddings/        # Text embedding
│   ├── rag/              # RAG implementation
│   ├── question_generation/ # Question creation
│   └── utils/            # Utility functions
├── results/              # Generated output
└── tests/               # Test files
```

## Error Handling

The system includes comprehensive error handling for common issues:

- PDF Processing Errors: Handles malformed PDFs and extraction issues
- LLM Connection Issues: Manages Ollama service disruptions
- Memory Management: Handles large documents through chunking
- Invalid Configurations: Validates all configuration parameters
- File System Errors: Manages file access and permissions issues

Errors are logged to `logs/app.log` with detailed information for troubleshooting.

## Troubleshooting

Common issues and solutions:

1. Ollama Connection Errors:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/generate
# Restart Ollama if needed
systemctl restart ollama  # Linux
```

2. Memory Issues:
- Adjust chunk_size in config.yaml
- Process smaller sections of the PDF
- Close unnecessary applications

3. PDF Processing Issues:
- Ensure PDF is text-based, not scanned
- Check file permissions
- Try different PDF formatting

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ollama team for providing local LLM capabilities
- Sentence Transformers for embedding models
- FAISS team for vector similarity search
- PyPDF2 for PDF processing capabilities

## Support

For issues and questions:
1. Check the existing issues on GitHub
2. Review the troubleshooting section
3. Create a new issue with detailed information about your problem
