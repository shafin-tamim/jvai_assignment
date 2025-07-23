from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
genai.configure(api_key=api_key)

# -------------------------
# PDF Loading and text extraction
# -------------------------
pdf_files = ["F:/Git/online_assingment/HSC26-Bangla1st-Paper.pdf"]  # replace with your actual PDF paths
combined_text = ""

for pdf_file in pdf_files:
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        combined_text += page.extract_text()

# -------------------------
# Split text into chunks
# -------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
text_chunks = splitter.split_text(combined_text)

# -------------------------
# Create FAISS vector store
# -------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
vector_store.save_local("faiss_index")

# -------------------------
# Load vector store for retrieval
# -------------------------
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# -------------------------
# Setup Gemini QA chain
# -------------------------
prompt_template = """You are a helpful assistant. Answer the question using only the provided context.
Answer in Bangla if the question is in Bangla, and in English if the question is in English.
Context:
{context}
Question:
{question}
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
qa_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

# -------------------------
# Convert the QA functionality into a function
# -------------------------
def get_answer(user_question):
    relevant_docs = vector_store.similarity_search(user_question)
    result = qa_chain({"input_documents": relevant_docs, "question": user_question}, return_only_outputs=True)
    return result["output_text"]

# Remove the direct question execution
if __name__ == "__main__":
    print("QA System initialized and ready to use")
