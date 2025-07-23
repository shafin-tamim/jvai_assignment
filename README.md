# HSC Question Answering System

An AI-powered question answering system specifically designed for HSC (Higher Secondary Certificate) content, supporting both Bangla and English queries.

## Setup Guide

1. Clone the repository:
```bash
git clone https://github.com/yourusername/online_assingment.git
cd online_assingment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run streamlit_app.py
```

## Tools & Libraries Used

- **Streamlit**: Web interface framework
- **LangChain**: Framework for building LLM applications
- **Google Gemini**: Large Language Model for text generation
- **FAISS**: Vector store for efficient similarity search
- **PyPDF2**: PDF processing
- **Sentence Transformers**: Text embeddings
- **Python-dotenv**: Environment variable management

## Key Features

- PDF document processing
- Vector-based similarity search
- Bilingual support (Bangla & English)
- Chat history maintenance
- Interactive web interface

## Sample Queries and Outputs

### Bangla Queries

Q: "বাংলা প্রথম পত্রে কয়টি গদ্য রচনা আছে?"
A: [Sample answer in Bangla]

Q: "রবীন্দ্রনাথ ঠাকুরের কোন কবিতাটি পাঠ্যসূচিতে অন্তর্ভুক্ত আছে?"
A: [Sample answer in Bangla]

### English Queries

Q: "How many poems are there in the first paper?"
A: [Sample answer in English]

Q: "What are the main themes in the curriculum?"
A: [Sample answer in English]

## API Documentation

### Main Functions

1. `get_answer(user_question: str) -> str`
   - Input: User question (string)
   - Output: AI-generated answer (string)
   - Supports both Bangla and English queries

### Vector Store Operations

- `vector_store.similarity_search(query)`
  - Performs semantic search on the document
  - Returns relevant document chunks

### Environment Variables

- `GOOGLE_API_KEY`: Required for Google Gemini API access

## Limitations

- Currently supports only PDF format documents
- Requires active internet connection
- Response time may vary based on query complexity

## Future Improvements

- Support for multiple document formats
- Offline mode support
- Enhanced context understanding
- Multi-document querying capability

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license]