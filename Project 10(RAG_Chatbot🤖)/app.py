import os
import tempfile

import streamlit as st

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Star Health AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
max-width:1100px;
}

.stChatMessage{
border-radius:15px;
padding:12px;
background:#f8f9fb;
margin-bottom:12px;
}

.title{
font-size:40px;
font-weight:700;
color:#0F172A;
}

.subtitle{
font-size:18px;
color:#64748B;
margin-bottom:25px;
}

.upload-box{
padding:15px;
border-radius:15px;
background:#F8FAFC;
border:1px solid #E2E8F0;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
"""
<div class='title'>
🩺 Star Health Insurance AI
</div>

<div class='subtitle'>
Retrieval-Augmented Chatbot powered by GPT-4o-mini
</div>
""",
unsafe_allow_html=True
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("⚙ Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your ChatGPT API key"
    )

    uploaded_file = st.file_uploader(
        "Upload starhealth.html",
        type=["html","htm"]
    )

    build_button = st.button(
        "Build Knowledge Base",
        use_container_width=True
    )

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# Prompt Template
# --------------------------------------------------

PROMPT = ChatPromptTemplate.from_template(
"""
You are an expert Star Health Insurance assistant.

Answer ONLY using the supplied context.

If the answer is not available inside the context,
reply:

"I couldn't find this information in the uploaded document."

Keep answers accurate and concise.

Question:
{question}

Context:
{context}

Answer:
"""
)
# --------------------------------------------------
# Build Knowledge Base
# --------------------------------------------------

if build_button:

    if not api_key:
        st.error("Please enter your OpenAI API Key.")
        st.stop()

    if uploaded_file is None:
        st.error("Please upload the Star Health HTML file.")
        st.stop()

    with st.spinner("Building knowledge base..."):

        try:

            os.environ["OPENAI_API_KEY"] = api_key

            # Save uploaded HTML temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".html"
            ) as tmp:

                tmp.write(uploaded_file.read())
                html_path = tmp.name

            # Load HTML
            loader = UnstructuredHTMLLoader(html_path)
            documents = loader.load()

            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)

            # Embedding Model
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key
            )

            # Vector Database
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 4
                }
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever

            st.success("Knowledge Base Ready!")

        except Exception as e:

            st.exception(e)

# --------------------------------------------------
# Chat Model
# --------------------------------------------------

def load_llm(api_key):

    return ChatOpenAI(

        model="gpt-4o-mini",

        temperature=0,

        api_key=api_key

    )


# --------------------------------------------------
# Create RAG Chain
# --------------------------------------------------

def create_chain(api_key):

    llm = load_llm(api_key)

    chain = (

        {
            "context": st.session_state.retriever,

            "question": RunnablePassthrough()

        }

        | PROMPT

        | llm

    )

    return chain


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

def ask_question(question):

    chain = create_chain(api_key)

    response = chain.invoke(question)

    return response.content


# --------------------------------------------------
# Ready Status
# --------------------------------------------------

if st.session_state.retriever is None:

    st.info("👈 Upload your Star Health HTML document and click **Build Knowledge Base**.")

    st.stop()
    # --------------------------------------------------
# Chat Interface
# --------------------------------------------------

st.divider()

st.subheader("💬 Ask Star Health AI")

# Clear Chat Button
col1, col2 = st.columns([5, 1])

with col2:
    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Display Previous Messages
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
question = st.chat_input(
    "Ask anything about Star Health Insurance..."
)

if question:

    # Save User Message
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                chain = create_chain(api_key)

                # Retrieve Context
                docs = st.session_state.retriever.invoke(question)

                context_text = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )

                answer = chain.invoke(question)

                response = answer.content

                st.markdown(response)

                # Save Assistant Reply
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

                # Expandable Sources
                with st.expander("📄 Retrieved Context"):

                    for i, doc in enumerate(docs, start=1):

                        st.markdown(f"### Chunk {i}")

                        st.write(doc.page_content)

                        st.divider()

            except Exception as e:

                st.error(str(e))

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "🩺 Star Health Insurance AI • Powered by GPT-4o-mini • LangChain • Chroma"
)
# ==================================================
# PROFESSIONAL DASHBOARD
# ==================================================

with st.sidebar:

    st.divider()

    st.subheader("📊 Dashboard")

    if st.session_state.retriever:

        st.success("Knowledge Base Loaded")

        st.metric(
            "Messages",
            len(st.session_state.chat_history)
        )

        if st.button(
            "Download Chat",
            use_container_width=True
        ):

            chat = ""

            for msg in st.session_state.chat_history:

                chat += (
                    f"{msg['role'].upper()}:\n"
                    f"{msg['content']}\n\n"
                )

            st.download_button(
                label="📄 Save Conversation",
                data=chat,
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True
            )

    st.divider()

    st.subheader("💡 Example Questions")

    st.markdown("""
- What is Family Floater Insurance?

- Explain Senior Citizen Plan.

- What are maternity benefits?

- What are the waiting periods?

- How can I buy health insurance?

- What is Critical Illness Cover?

- Compare individual and family plans.

- Which policy is best for young adults?
""")

# ==================================================
# KNOWLEDGE BASE INFO
# ==================================================

if st.session_state.retriever:

    with st.expander("📚 Knowledge Base"):

        try:

            docs = st.session_state.vectorstore.get()

            st.metric(
                "Document Chunks",
                len(docs["documents"])
            )

        except:

            st.info("Knowledge Base Ready.")

# ==================================================
# CHAT STATISTICS
# ==================================================

if len(st.session_state.chat_history) > 0:

    user_messages = len(
        [
            x for x in st.session_state.chat_history
            if x["role"] == "user"
        ]
    )

    assistant_messages = len(
        [
            x for x in st.session_state.chat_history
            if x["role"] == "assistant"
        ]
    )

    with st.expander("📈 Conversation Statistics"):

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Questions",
                user_messages
            )

        with col2:

            st.metric(
                "Responses",
                assistant_messages
            )

# ==================================================
# ABOUT
# ==================================================

with st.expander("ℹ About"):

    st.write("""
This chatbot uses:

• GPT-4o-mini

• LangChain

• Chroma Vector Database

• Recursive Text Splitting

• OpenAI Embeddings

• Retrieval-Augmented Generation (RAG)

Upload any Star Health Insurance HTML document and ask questions from it.
""")

# ==================================================
# PREMIUM FOOTER
# ==================================================

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;
font-size:14px;
color:gray">

🩺 Star Health AI Assistant

Powered by OpenAI GPT-4o-mini • LangChain • ChromaDB

Built with ❤️ using Streamlit

</div>
""",
unsafe_allow_html=True
)
# ============================================================
# PART 5 - PREMIUM FEATURES
# ============================================================

import time
from datetime import datetime

# -----------------------------
# Theme Toggle
# -----------------------------

with st.sidebar:

    st.divider()

    dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=False
    )

if dark_mode:

    st.markdown("""
    <style>

    .stApp{
        background:#0E1117;
        color:white;
    }

    .stChatMessage{
        background:#1E293B;
        border-radius:16px;
        padding:12px;
    }

    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Session Info
# -----------------------------

if "session_start" not in st.session_state:

    st.session_state.session_start = datetime.now()

# -----------------------------
# Response Streaming
# -----------------------------

def stream_text(text):

    placeholder = st.empty()

    output = ""

    for word in text.split():

        output += word + " "

        placeholder.markdown(output + "▌")

        time.sleep(0.02)

    placeholder.markdown(output)

# -----------------------------
# Replace Existing Response
# -----------------------------
# Wherever you have:

# st.markdown(response)

# Replace with:

# stream_text(response)

# -----------------------------
# Copy Button
# -----------------------------

if len(st.session_state.chat_history):

    last = st.session_state.chat_history[-1]

    if last["role"] == "assistant":

        st.code(last["content"])

# -----------------------------
# Conversation Timer
# -----------------------------

elapsed = datetime.now() - st.session_state.session_start

mins = elapsed.seconds // 60

st.sidebar.metric(
    "Session Time",
    f"{mins} min"
)

# -----------------------------
# Conversation Export
# -----------------------------

if st.sidebar.button("💾 Export Markdown"):

    md = "# Chat History\n\n"

    for msg in st.session_state.chat_history:

        md += f"## {msg['role']}\n"

        md += msg["content"]

        md += "\n\n"

    st.download_button(

        "Download",

        md,

        file_name="conversation.md",

        mime="text/markdown"

    )

# -----------------------------
# Search Conversation
# -----------------------------

search = st.sidebar.text_input(

    "🔍 Search Chat"

)

if search:

    st.subheader("Search Results")

    for msg in st.session_state.chat_history:

        if search.lower() in msg["content"].lower():

            st.write(f"**{msg['role']}**")

            st.write(msg["content"])

            st.divider()

# -----------------------------
# Keyboard Shortcuts
# -----------------------------

st.sidebar.info("""
### Shortcuts

Enter → Send

Ctrl+L → Clear Chat

Upload HTML then Build KB

Ask Questions
""")

# -----------------------------
# Welcome Screen
# -----------------------------

if len(st.session_state.chat_history) == 0:

    st.markdown("""

## 👋 Welcome

You can ask things like:

• What are the benefits of Family Floater Insurance?

• Explain Critical Illness Cover.

• Which plan is best for senior citizens?

• Compare Individual vs Family plans.

• What is Maternity Health Insurance?

""")
    