# HSC Bangla Question Answering System

An intelligent QA system built to answer questions about HSC Bangla content using Google's Gemini AI.

## Setup Guide

1. Clone the repository
```bash
git clone <repository-url>
cd 10ms_assignment
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your Google API key
```bash
GOOGLE_API_KEY=your_api_key_here
```

4. Run the application
```bash
streamlit run app.py
```

## Tools & Technologies

- **Framework**: Streamlit
- **Language Model**: Google Gemini 2.5 Flash
- **Vector Store**: FAISS
- **Embeddings**: Google Generative AI Embeddings
- **PDF Processing**: PyPDF2
- **Text Processing**: LangChain

## Key Features

- Bilingual support (Bangla & English)
- PDF document processing
- Vector similarity search
- Conversation history
- User-friendly interface

## Sample Queries & Outputs

### Bangla Queries
Q: "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
A: শুম্ভুনাথ

Q: "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"
A: মামাকে

Q: "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
A: ১৫ বছর

### Sample Screenshots

#### Main Interface
![Main Interface](./assets/image.png)
*The main application interface showing the question input area and response section*

#### Sample Question-Answer
!(./assets/image-1.png)

#### PDF Processing
!(./assets/image-2.png)


## System Architecture

1. **Document Processing**
   - PDF text extraction
   - Text chunking
   - Embedding generation

2. **Query Processing**
   - Question embedding
   - Similarity search
   - Context retrieval
   - Answer generation

## Limitations & Future Improvements

- Currently supports single PDF document
- Limited to text-based QA
- Future scope:
  - Multiple document support
  - Image-based questions
  - Enhanced context understanding
  - API integration

## Performance Metrics

- Average response time: ~2-3 seconds
- Context retrieval accuracy: Based on vector similarity
- Answer relevance: Determined by Gemini model
- Memory usage: ~500MB-1GB depending on document size

## License

[Add your license information]
