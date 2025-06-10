# Photography AI Assistant

An AI-powered photography assistant that uses RAG (Retrieval Augmented Generation) architecture to answer photography-related questions. What I have not chechecked in here are random PDF files on teaching photography that I found on the web, and dropped it in there. The llama3/Ollama/sentence-transformers based AI is then able to access the documentation and provide answers needed.
![screenshot-document-ai](https://github.com/user-attachments/assets/0caf0fd1-828e-41bd-8027-49cec8f55298)

## Features

- Interactive Q&A about photography topics
- Local document processing and storage
- Efficient document retrieval using vector similarity
- Contextual responses using RAG architecture

## Requirements

### Core Dependencies
- Python 3.8+
- Ollama (for local LLM inference)
- LangChain and LangChain Community (for RAG pipeline orchestration)
- ChromaDB (for vector storage)
- HuggingFace Transformers (for embeddings)

### Python Packages
```bash
langchain>=0.1.0
langchain-community>=0.0.10
chromadb>=0.4.22
ollama>=0.1.6
sentence-transformers>=2.2.2
pypdf>=3.17.1
unstructured>=0.11.0
python-magic>=0.4.27
python-magic-bin>=0.4.14  # For macOS
```

### System Requirements
- Memory: Minimum 8GB RAM (16GB recommended)
- Storage: 1GB for base installation, additional space for knowledge base
- CPU: Modern multi-core processor
- GPU: Optional, but recommended for faster embedding generation

### Optional Dependencies
- `unstructured` and related packages for handling various document formats:
  - docx
  - markdown
  - html
- `tiktoken` for token counting
- `faiss-cpu` or `faiss-gpu` for alternative vector storage

### Development Tools
- Git (for version control)
- Virtual environment manager (venv or conda)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Pras-subash/photo-ai-assistant.git
cd photography-ai-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run Ollama (ensure it's installed first):
```bash
ollama run llama3
```

## Usage

Run the main script:
Please note , you might want to create a venv where you do this.
```bash
python photography_ai.py
```

The assistant will load the knowledge base and prompt you for questions about photography.

## License

MIT License
