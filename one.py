import streamlit as st
import os
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

def main():
    st.title("Groq Chat App")

    # Add customization options to the sidebar
    st.sidebar.title('Select an LLM')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length)

    # PDF Upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.sidebar.markdown("**PDF successfully uploaded!**")

        # Process uploaded PDF file
        pdf_contents = uploaded_file.read()
        pdf_file = BytesIO(pdf_contents)
        pdf_reader = PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)

        st.sidebar.write(f"Number of pages in uploaded PDF: {num_pages}")

        # Display PDF content in the sidebar
        st.sidebar.markdown("**PDF Content:**")
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            st.sidebar.write(f"Page {page_num + 1}:")
            st.sidebar.write(text)

        # Initialize Groq Langchain chat object and conversation
        groq_chat = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model
        )

        conversation = ConversationChain(
            llm=groq_chat,
            memory=memory
        )

        # User interaction section
        st.header("Ask a Question about the Uploaded PDF")

        user_question = st.text_area("Ask a question:")

        if user_question:
            response = conversation(user_question)
            message = {'human': user_question, 'AI': response['response']}
            st.write("Chatbot:", response['response'])
            st.session_state.chat_history.append(message)

if __name__ == "__main__":
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    main()
