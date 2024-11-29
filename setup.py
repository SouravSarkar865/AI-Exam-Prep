import os
import platform
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Verify Python version is 3.9 or higher."""
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required")
        sys.exit(1)

def install_python_requirements():
    """Install Python package dependencies."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully installed Python dependencies")
    except subprocess.CalledProcessError:
        print("Error installing Python dependencies")
        sys.exit(1)

def install_ollama():
    """Install Ollama based on the operating system."""
    system = platform.system().lower()
    
    if system == "linux":
        try:
            print("Installing Ollama for Linux...")
            subprocess.check_call(["curl", "-fsSL", "https://ollama.com/install.sh", "|", "sh"], shell=True)
        except subprocess.CalledProcessError:
            print("Error installing Ollama. Please install manually from https://ollama.com/download")
            return False
    
    elif system == "darwin":  # macOS
        print("Please install Ollama for macOS manually:")
        print("1. Visit https://ollama.com/download")
        print("2. Download and run the macOS installer (.dmg file)")
        return False
    
    elif system == "windows":
        print("Please install Ollama for Windows manually:")
        print("1. Visit https://ollama.com/download")
        print("2. Download and run the Windows installer (.msi file)")
        return False
    
    else:
        print(f"Unsupported operating system: {system}")
        return False
    
    return True

def setup_project_structure():
    """Create necessary project directories."""
    directories = [
        "resources",
        "results",
        "logs",
        "data/vector_store"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Successfully created project directories")

def check_ollama_installation():
    """Check if Ollama is installed and running."""
    try:
        subprocess.check_call(["ollama", "--version"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    print("Starting project setup...")
    
    # Check Python version
    check_python_version()
    
    # Install Python dependencies
    print("\nInstalling Python dependencies...")
    install_python_requirements()
    
    # Check if Ollama is already installed
    print("\nChecking Ollama installation...")
    if not check_ollama_installation():
        print("Ollama not found. Attempting installation...")
        if install_ollama():
            print("Successfully installed Ollama")
        else:
            print("\nPlease install Ollama manually and run this setup script again")
            sys.exit(1)
    else:
        print("Ollama is already installed")
    
    # Create project structure
    print("\nSetting up project directories...")
    setup_project_structure()
    
    print("\nSetup complete! Next steps:")
    print("1. If you installed Ollama manually, start it according to your OS instructions")
    print("2. Place your Project Management PDF in the 'resources' directory")
    print("3. Run 'python src/main.py' to start generating questions")

if __name__ == "__main__":
    main()
