import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter

#
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from htmlTemplates import css, bot_template, user_template


# ---------------- PDF TEXT ---------------- #
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


# ---------------- TEXT CHUNKS ---------------- #
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


# ---------------- VECTOR STORE ---------------- #
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )


# ---------------- LLM + CHAIN ---------------- #
def get_conversation_chain(vectorstore):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


# ---------------- CHAT ---------------- #
def handle_userinput(user_question):
    with st.spinner("Thinking..."):
        response = st.session_state.conversation(
            {'question': user_question}
        )

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


# ---------------- MAIN ---------------- #
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    api_key = st.secrets["GOOGLE_API_KEY"]

    st.set_page_config(
        page_title="Chat with PDFs",
        page_icon="📚"
    )

    st.write(css, unsafe_allow_html=True)

    # init session
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # UI
    st.markdown("<h1 style='text-align: center;'>📚 Chat with your PDFs</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Ask anything from your documents</p>", unsafe_allow_html=True)

    user_question = st.text_input("💬 Ask something...")

    if user_question:
        if st.session_state.conversation is None:
            st.warning("⚠️ Please upload and process PDFs first.")
        else:
            handle_userinput(user_question)

    # Sidebar
    with st.sidebar:
        st.subheader("📂 Your documents")

        pdf_docs = st.file_uploader(
            "Upload PDFs",
            accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("Processing..."):

                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(
                    vectorstore
                )

                st.success("✅ Documents processed! You can now ask questions.")


if __name__ == '__main__':
    main()