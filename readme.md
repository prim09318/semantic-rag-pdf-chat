# Semantic RAG PDF Chat

A Retrieval-Augmented Generation (RAG) application that enables users to upload multiple PDF documents and interact with them through natural language conversations. The system uses semantic search with vector embeddings and Gemini 2.5 Flash to provide context-aware answers grounded in the uploaded documents.

## Features

* Upload and process multiple PDF documents
* Automatic text extraction and chunking
* Semantic search using Sentence Transformers embeddings
* FAISS vector database for efficient retrieval
* Conversational memory for follow-up questions
* Powered by Google Gemini 2.5 Flash
* Interactive Streamlit web interface
* Retrieval-Augmented Generation (RAG) pipeline

## Architecture

```text
PDF Documents
      │
      ▼
Text Extraction (PyPDF2)
      │
      ▼
Text Chunking
      │
      ▼
Sentence Transformers
(all-MiniLM-L6-v2)
      │
      ▼
FAISS Vector Store
      │
      ▼
Semantic Retrieval
      │
      ▼
Gemini 2.5 Flash
      │
      ▼
Conversational Answers
```

## Tech Stack

### Frontend

* Streamlit

### LLM

* Google Gemini 2.5 Flash

### Retrieval & Orchestration

* LangChain
* FAISS

### Embeddings

* Sentence Transformers
* all-MiniLM-L6-v2

### Document Processing

* PyPDF2

## Installation

### Clone the Repository

```bash
git clone https://github.com/<your-username>/semantic-rag-pdf-chat.git
cd semantic-rag-pdf-chat
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Generate your Gemini API key from Google AI Studio.

## Run the Application

```bash
streamlit run app.py
```

The application will launch in your browser at:

```text
http://localhost:8501
```

## Usage

1. Launch the application.
2. Upload one or more PDF documents.
3. Click **Process**.
4. Wait for embeddings and vector indexing to complete.
5. Ask questions about the uploaded documents.
6. Continue the conversation with follow-up questions.

## Example Questions

* What is the main topic discussed in these documents?
* Summarize the key findings.
* What recommendations are provided?
* Compare the conclusions across the uploaded PDFs.
* Explain this concept in simple terms.

## Project Structure

```text
semantic-rag-pdf-chat/
│
├── app.py
├── htmlTemplates.py
├── requirements.txt
├── .env
│
├── venv/
│
└── README.md
```

## Requirements

* Python 3.10
* Google Gemini API Key
* Internet connection for LLM inference

## Future Improvements

* Source citation support
* PDF page references in answers
* Chat history export
* Support for DOCX and TXT files
* Hybrid retrieval techniques
* Persistent vector database storage
* Streaming responses
* Multi-user support

## License

This project is intended for educational and portfolio purposes.

## Author

Developed as a semantic document question-answering system using Retrieval-Augmented Generation (RAG), LangChain, FAISS, Sentence Transformers, and Google Gemini.
