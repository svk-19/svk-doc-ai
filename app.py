import streamlit as st
import os
os.makedirs("data", exist_ok=True)

from datetime import datetime

from utils.document_loader import (
    load_document
)
from utils.chunking import split_text

from utils.embeddings import (
    create_embeddings,
    create_query_embedding
)

from utils.vector_store import (
    store_embeddings,
    retrieve_chunks
)

from utils.gemini_rag import (
    generate_answer,
    generate_pdf_summary
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="SVK Doc AI",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stApp {
    background-color: #0E1117;
}

.chat-user{
    background:#1E293B;
    padding:15px;
    border-radius:15px;
    margin-bottom:10px;
    border-left:5px solid #4ADE80;
}

.chat-ai{
    background:#161B22;
    padding:15px;
    border-radius:15px;
    margin-bottom:20px;
    border-left:5px solid #38BDF8;
}

.answer-box{
    background:#161B22;
    padding:20px;
    border-radius:15px;
    border:1px solid #4ADE80;
}

.big-title{
    text-align:center;
    font-size:55px;
    font-weight:bold;
    color:#4ADE80;
}

.sub-title{
    text-align:center;
    color:#9CA3AF;
    font-size:18px;
    margin-bottom:25px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

if "pdf_summary" not in st.session_state:
    st.session_state.pdf_summary = ""

if "document_summary" not in st.session_state:
    st.session_state.document_summary = ""

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🤖 SVK Doc AI")

    st.caption(
        "Professional AI Document Assistant"
    )

    st.divider()

    st.subheader("🚀 Features")

    st.success("📄 Document Chat")
    st.success("🧠 Semantic Search")
    st.success("🤖 Gemini AI")
    st.success("📚 RAG Pipeline")
    st.success("💬 Memory")
    st.success("📥 Export Chat")
    st.success("📝 Document Summary")

    st.divider()

    st.subheader("⚙️ AI Stack")

    st.info("LLM : Gemini 2.5 Flash")
    st.info("Embedding : MiniLM-L6-v2")
    st.info("Vector DB : ChromaDB")

    st.divider()

    st.subheader("👨‍💻 Developer")

    st.write("Vamshi Krishna")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🆕 New Chat"):

            st.session_state.chat_history = []
            st.rerun()

    with col2:
        if st.button("🗑️ Clear"):

            st.session_state.chat_history = []
            st.rerun()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="big-title">
    🤖 SVK Doc AI
    </div>

    <div class="sub-title">
    Turn Document's into Searchable Knowledge with AI
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload Document",
    type=[
        "pdf",
        "docx",
        "txt"
    ]
)

if uploaded_file:

    file_extension = (
        uploaded_file.name
        .split(".")[-1]
        .lower()
    )

    file_path = (
        f"data/uploaded.{file_extension}"
    )

    with open("data/uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(
        f"✅ {file_extension.upper()} Uploaded Successfully"
    )

    text = load_document(
        file_path,
        file_extension
    )

    chunks = split_text(text)

    embeddings = create_embeddings(
        chunks
    )

    store_embeddings(
        chunks,
        embeddings
    )

    st.success(
        f"✅ Document Processed ({len(chunks)} Chunks)"
    )

    st.session_state.pdf_loaded = True

    # ------------------------------------------
    # SUMMARY BUTTON
    # ------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "📝 Generate Document Summary"
        ):

            with st.spinner(
                "Creating Summary..."
            ):

                summary = generate_pdf_summary(
                    text[:15000]
                )

                st.session_state.document_summary = summary

    with col2:

        if st.button(
            "📥 Export Chat"
        ):

            export_text = ""

            for chat in st.session_state.chat_history:

                export_text += (
                    f"\nQUESTION:\n"
                    f"{chat['question']}\n\n"
                    f"ANSWER:\n"
                    f"{chat['answer']}\n"
                    f"{'-'*50}\n"
                )

            st.download_button(
                label="Download TXT",
                data=export_text,
                file_name="svk_doc_ai_chat.txt",
                mime="text/plain"
            )

    # ------------------------------------------
    # SUMMARY
    # ------------------------------------------

    if st.session_state.document_summary:

        st.subheader(
            "📄 Document Summary"
        )

        st.markdown(
            f"""
            <div class="answer-box">
            {st.session_state.pdf_summary}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # ------------------------------------------
    # CHAT INPUT
    # ------------------------------------------

    question = st.chat_input(
        "Ask Anything About Your PDF..."
    )

    if question:

        query_embedding = (
            create_query_embedding(
                question
            )
        )

        results = retrieve_chunks(
            query_embedding
        )

        retrieved_chunks = (
            results["documents"][0]
        )

    


        # --------------------------------------
        # MEMORY CONTEXT
        # --------------------------------------

        memory_context = ""

        last_chats = (
            st.session_state.chat_history[-3:]
        )

        for chat in last_chats:

            memory_context += (
                f"User: {chat['question']}\n"
                f"Assistant: {chat['answer']}\n\n"
            )

        with st.spinner(
            "🤖 Thinking..."
        ):

            answer = generate_answer(
                question,
                retrieved_chunks,
                memory_context
            )

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "time": datetime.now().strftime(
                    "%H:%M"
                )
            }
        )

        st.rerun()

# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if st.session_state.chat_history:

    st.subheader(
        "💬 Conversation"
    )

    for chat in st.session_state.chat_history:

        st.markdown(
            f"""
            <div class="chat-user">
            🧑 <b>You</b><br><br>
            {chat['question']}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="chat-ai">
            🤖 <b>SVK Doc AI</b><br><br>
            {chat['answer']}
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# EMPTY SCREEN
# --------------------------------------------------

if not uploaded_file:

    st.info(
        "👆 Upload a PDF to start chatting."
    )

    