# Financial Policy Question Answering System

An intelligent QA system built to analyze and answer questions about financial policy documents using Google's Gemini AI.

## Features

- Context-aware responses from policy documents
- Conversation history tracking
- Source citation in answers
- Vector similarity search
- User-friendly interface
- PDF document processing

## Setup

1. Clone the repository
```bash
git clone <repository-url>
cd jvai
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure environment
- Create a `.env` file
- Add your Google API key:
```bash
GOOGLE_API_KEY=your_api_key_here
```

4. Launch application
```bash
streamlit run app.py
```

## Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini 2.5 Flash
- **Vector Store**: FAISS
- **Embeddings**: Google Generative AI Embeddings
- **Document Processing**: PyPDF2
- **Framework**: LangChain

## Architecture

### Document Processing Pipeline
1. PDF text extraction using PyPDF2
2. Text chunking (10,000 chars with 1,000 overlap)
3. Embedding generation using Google's model
4. FAISS vector store indexing

### Query Processing Flow
1. Question embedding
2. Similarity search in FAISS
3. Context retrieval
4. Answer generation with source citation

## System Features

- **Contextual QA**: Answers based on document content
- **Source Citations**: References to specific sections
- **History Tracking**: Maintains conversation context
- **Vector Search**: Efficient document retrieval
- **Local Processing**: No external API calls except Gemini

## Performance

- Response Time: ~2-3 seconds
- Memory Usage: 500MB-1GB
- Context Window: 10,000 characters
- Overlap: 1,000 characters

## Future Improvements

- Multi-document support
- Enhanced context understanding
- API endpoint creation
- Performance optimization
- Document preprocessing improvements

## Performance Metrics

- Average response time: ~2-3 seconds
- Context retrieval accuracy: Based on vector similarity
- Answer relevance: Determined by Gemini model
- Memory usage: ~500MB-1GB depending on document size

## License


"# jvai_assignment" 
