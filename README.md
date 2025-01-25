# PDF Question-Answering System with Local LLM

A Retrieval-Augmented Generation (RAG) system that allows users to upload PDFs and ask questions about their content. The system uses TF-IDF vectorization with NLTK preprocessing for efficient document retrieval and Ollama for local LLM inference.

## Features

- **PDF Processing**: Upload and process multiple PDF documents
- **Advanced Text Processing**: NLTK-based preprocessing including lemmatization and stopword removal
- **Efficient Retrieval**: TF-IDF vectorization with cosine similarity search
- **Local LLM Integration**: Uses Ollama for generating responses without external API calls
- **Interactive UI**: Clean Gradio interface with document preview
- **Source Attribution**: Shows source documents and similarity scores for transparency

## Technical Stack

- **Frontend**: Gradio
- **Document Processing**: LangChain, PyPDF
- **Text Processing**: NLTK
- **Vector Embedding**: scikit-learn (TF-IDF)
- **LLM Integration**: Ollama
- **Storage**: Local file system with pickle serialization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/roshnn24/FASTACK.AI-RAG-Doc-System.git
cd pdf-qa-system
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama:
```bash
# On macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# For Windows, visit: https://ollama.com/download/windows
```

## Usage

1. Start Ollama server:
```bash
ollama serve
```

2. Pull the required model:
```bash
ollama pull llama3.1:8b
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:7860
```

## How It Works

1. **Document Processing Pipeline**:
   - PDF chunking for efficient processing
   - NLTK preprocessing (lemmatization, stopword removal)
   - TF-IDF vectorization for numerical representation

2. **Query Processing**:
   - User questions undergo same preprocessing
   - Cosine similarity finds relevant document chunks
   - Top chunks sent to LLM for answer generation

3. **Response Generation**:
   - Local LLM generates answers from context
   - System provides source attribution
   - Displays similarity scores for transparency

## Project Structure

```
pdf-qa-system/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── data/              # Stored PDF files
└── vector_db/            # Vector store data
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LangChain for document processing utilities
- Gradio team for the UI framework
- Ollama team for the local LLM capability

---
Created by [Roshaun Infant R] - Feel free to connect!
