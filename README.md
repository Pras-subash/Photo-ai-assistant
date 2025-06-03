# Photography AI Assistant

An AI-powered photography assistant that uses RAG (Retrieval Augmented Generation) architecture to answer photography-related questions. What I have not chechecked in here are random PDF files on teaching photography that I found on the web, and dropped it in there. The llama3/Ollama/sentence-transformers based AI is then able to access the documentation and provide answers needed.

## Features

- Interactive Q&A about photography topics
- Local document processing and storage
- Efficient document retrieval using vector similarity
- Contextual responses using RAG architecture

## Requirements

- Python 3.8+
- Ollama
- LangChain
- ChromaDB
- HuggingFace Transformers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Pras-subash/photography-ai-assistant.git
cd photography-ai-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run Ollama (ensure it's installed first):
```bash
ollama run llama2
```

## Usage

Run the main script:
```bash
python photography_ai.py
```

The assistant will load the knowledge base and prompt you for questions about photography.

## License

MIT License
