#!/bin/bash

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Ollama installation...${NC}"

# Detect operating system
OS="$(uname -s)"

case "${OS}" in
    Linux*)
        echo -e "${YELLOW}Detected Linux system. Installing Ollama...${NC}"
        if ! command -v curl &> /dev/null; then
            echo -e "${RED}curl is not installed. Installing curl...${NC}"
            sudo apt-get update && sudo apt-get install -y curl
        fi
        
        echo "Downloading and running Ollama installer..."
        curl https://ollama.ai/install.sh | sh
        
        # Verify installation
        if command -v ollama &> /dev/null; then
            echo -e "${GREEN}Ollama installed successfully!${NC}"
            
            echo -e "${YELLOW}Pulling Mixtral model (this might take a while)...${NC}"
            ollama pull mixtral
            
            echo -e "${GREEN}Installation complete! Ollama is ready to use.${NC}"
        else
            echo -e "${RED}Installation failed. Please try manual installation from https://ollama.com/download${NC}"
            exit 1
        fi
        ;;
        
    Darwin*)
        echo -e "${YELLOW}Detected macOS system.${NC}"
        echo -e "${YELLOW}Please install Ollama manually:${NC}"
        echo "1. Visit https://ollama.com/download"
        echo "2. Download the macOS installer (.dmg file)"
        echo "3. Open the downloaded file and follow installation instructions"
        echo "4. After installation, run: ollama pull mixtral"
        exit 0
        ;;
        
    MINGW*|CYGWIN*|MSYS*)
        echo -e "${YELLOW}Detected Windows system.${NC}"
        echo -e "${YELLOW}Please install Ollama manually:${NC}"
        echo "1. Visit https://ollama.com/download"
        echo "2. Download the Windows installer (.msi file)"
        echo "3. Run the installer and follow installation steps"
        echo "4. After installation, run: ollama pull mixtral"
        exit 0
        ;;
        
    *)
        echo -e "${RED}Unsupported operating system: ${OS}${NC}"
        exit 1
        ;;
esac
