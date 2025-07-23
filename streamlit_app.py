import streamlit as st
from app import get_answer

# Initialize session state for storing chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("HSC Question Answering System")

# Create input field and button
user_question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if user_question:
        # Get answer using the QA system
        answer = get_answer(user_question)
        
        # Add to chat history
        st.session_state.chat_history.append({"question": user_question, "answer": answer})
        
        # Show the current answer
        st.write("Answer:")
        st.write(answer)

# Display chat history
if st.session_state.chat_history:
    st.subheader("Recent Questions and Answers")
    for i, qa in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5 QA pairs
        with st.expander(f"Q: {qa['question']}", expanded=False):
            st.write("A:", qa['answer'])
