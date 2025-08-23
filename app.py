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
pdf_files = ["F:/Git/jvai/For Task - Policy file.pdf"]
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
prompt_template = """
You are a helpful AI chatbot built to answer questions about a financial policy document.
Use ONLY the provided context from the document. 
Do not use outside knowledge. 
Always include the page or section number from the source in your answer. 
If the answer cannot be found in the context, say: "I could not find this information in the policy document."

Conversation so far:
{history}

Context from document:
{context}

User Question:
{question}

Answer (clear, concise, and with source citation):
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
qa_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

# -------------------------
# Function to get answer
# -------------------------
def get_answer(user_question, history=""):
    relevant_docs = vector_store.similarity_search(user_question)
    result = qa_chain(
        {
            "input_documents": relevant_docs, 
            "question": user_question,
            "history": history,
            "context": "\n".join(doc.page_content for doc in relevant_docs)
        }, 
        return_only_outputs=True
    )
    return result["output_text"]

# -------------------------
# Streamlit Interface
# -------------------------
import streamlit as st

st.set_page_config(page_title="Financial Policy QA System", page_icon="💰", layout="wide")

# Sidebar for chat history
with st.sidebar:
    st.header("💬 Chat History")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if st.session_state.chat_history:
        for i, qa in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {qa['question']}", expanded=False):
                st.markdown(f"A: {qa['answer']}")

# Main content
st.title("💰 Financial Policy Question Answering System")

# Input field
user_question = st.text_input("Ask a question about the Financial Policy Document:")

if st.button("Get Answer"):
    if user_question.strip() != "":
        # Get previous chat history as formatted string
        history = "\n".join([
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in st.session_state.chat_history
        ])
        
        # Call your retrieval + Gemini chatbot function with history
        answer = get_answer(user_question, history)
        st.session_state.chat_history.append({"question": user_question, "answer": answer})
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("⚠️ Please enter a question before pressing the button.")
else:
    st.warning("⚠️ Please enter a question before pressing the button.")
