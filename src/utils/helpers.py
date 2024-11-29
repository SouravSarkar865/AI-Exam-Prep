import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

def setup_logging(logging_config: Dict) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        logging_config (Dict): Logging configuration settings
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(os.path.dirname(logging_config['file_path']))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, logging_config['level']),
        format=logging_config['format'],
        handlers=[
            logging.FileHandler(logging_config['file_path']),
            logging.StreamHandler()  # Also log to console
        ]
    )

def save_json(data: Any, filepath: str, pretty: bool = True) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath (str): Path to save the JSON file
        pretty (bool): Whether to format JSON with indentation
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=4, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
                
    except Exception as e:
        logging.error(f"Error saving JSON to {filepath}: {str(e)}")
        raise IOError(f"Failed to save JSON file: {str(e)}")

def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        Any: Loaded JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON file not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in file {filepath}: {str(e)}")
        raise

def ensure_directory(directory: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory (str): Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        str: Current timestamp
    """
    return datetime.utcnow().isoformat() + "Z"

def validate_file_path(filepath: str, create_dir: bool = False) -> bool:
    """
    Validate if a file path is valid and optionally create directory.
    
    Args:
        filepath (str): File path to validate
        create_dir (bool): Whether to create directory if it doesn't exist
        
    Returns:
        bool: True if path is valid
    """
    try:
        directory = os.path.dirname(filepath)
        if create_dir and directory:
            os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Invalid file path {filepath}: {str(e)}")
        return False

def chunk_list(lst: list, chunk_size: int) -> list:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst (list): List to chunk
        chunk_size (int): Size of each chunk
        
    Returns:
        list: List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def format_error(error: Exception) -> Dict:
    """
    Format an exception into a structured error dictionary.
    
    Args:
        error (Exception): Exception to format
        
    Returns:
        Dict: Formatted error information
    """
    return {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': get_current_timestamp()
    }

def clean_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Clean and normalize text.
    
    Args:
        text (str): Text to clean
        max_length (Optional[int]): Maximum length of text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
        
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove control characters
    text = ''.join(char for char in text if char.isprintable() or char == '\n')
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + '...'
    
    return text.strip()

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        Dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        raise

def validate_config(config: Dict, required_fields: list) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config (Dict): Configuration to validate
        required_fields (list): List of required field paths
        
    Returns:
        bool: True if config is valid
    """
    def get_nested_value(d: Dict, path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = path.split('.')
        value = d
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value
    
    # Check all required fields exist and have non-empty values
    for field in required_fields:
        value = get_nested_value(config, field)
        if value is None or (isinstance(value, (str, list, dict)) and not value):
            logging.error(f"Missing or empty required config field: {field}")
            return False
    
    return True

def setup_project_directories(base_dirs: list) -> None:
    """
    Set up project directory structure.
    
    Args:
        base_dirs (list): List of directory paths to create
    """
    for directory in base_dirs:
        ensure_directory(directory)
        logging.info(f"Created directory: {directory}")
