import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st

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
pdf_files = ["F:/Git/online_assingment/HSC26-Bangla1st-Paper.pdf"]
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
# Create or Load FAISS vector store
# -------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
if not os.path.exists("faiss_index/index.faiss"):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
else:
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
# Function to get answer
# -------------------------
def get_answer(user_question):
    relevant_docs = vector_store.similarity_search(user_question)
    result = qa_chain({"input_documents": relevant_docs, "question": user_question}, return_only_outputs=True)
    return result["output_text"]

# -------------------------
# Streamlit Interface
# -------------------------
st.set_page_config(page_title="HSC Bangla QA System", page_icon="📘", layout="centered")
st.title("📘 HSC Bangla Question Answering System")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field
user_question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_question.strip() != "":
        answer = get_answer(user_question)
        st.session_state.chat_history.append({"question": user_question, "answer": answer})
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question before pressing the button.")

# Display chat history
if st.session_state.chat_history:
    st.subheader("🕑 Recent Questions and Answers")
    for i, qa in enumerate(reversed(st.session_state.chat_history[-5:])):
        with st.expander(f"Q: {qa['question']}", expanded=False):
            st.write("A:", qa['answer'])
