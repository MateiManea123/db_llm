import os
import uuid
import dotenv
import streamlit as st

from db_methods import stream_db_response, has_db, load_db_generic


from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

# ---------- ENV & CONFIG ----------

dotenv.load_dotenv()

AZURE_API_KEY = os.getenv("AZ_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZ_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = "gpt-4o"  # numele deployment-ului tău din Azure portal

if not AZURE_API_KEY or not AZURE_ENDPOINT:
    st.error(
        "AZ_OPENAI_API_KEY sau AZ_OPENAI_ENDPOINT nu sunt setate în .env.\n"
        "Adaugă-le și repornește aplicația."
    )
    st.stop()

# un singur model: deployment-ul tău
MODELS = [f"azure-openai/{AZURE_DEPLOYMENT}"]
st.session_state.model = f"azure-openai/{AZURE_DEPLOYMENT}"

st.set_page_config(
    page_title="DB LLM",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------- INITIAL SESSION STATE ----------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"},
    ]

# ---------- HEADER ----------

st.html(
    """<h2 style="text-align: center;">📚 <i> DB LLM </i> 💬</h2>"""
)

# ---------- SIDEBAR: MODEL + RAG SOURCES ----------
with st.sidebar:

    st.divider()

    # adăugăm și modul DB Assistant
    mode = st.radio(
        "Mode",
        options=["DB Assistant"],
        index=0,
        key="chat_mode",
    )



    cols0 = st.columns(2)
    with cols0[0]:
        is_vector_db_loaded = (
            "vector_db" in st.session_state
            and st.session_state.vector_db is not None
        )
        st.toggle(
            "Use RAG",
            value=(mode == "RAG" and is_vector_db_loaded),
            key="use_rag",
            disabled=(mode != "RAG" or not is_vector_db_loaded),
        )

    with cols0[1]:
        st.button(
            "Clear Chat",
            on_click=lambda: st.session_state.messages.clear(),
            type="primary",
        )

    st.header("RAG Sources:")

    # File upload input for RAG with documents
    st.file_uploader(
        "📄 Upload a document",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        on_change=load_doc_to_db,
        key="rag_docs",
    )
    st.header("🗄️ Database (SQL Server from .bak)")

    st.file_uploader(
        "Upload .bak (SQL Server backup)",
        type=["bak"],
        key="db_file",
        on_change=load_db_generic,
    )

    if has_db():
        st.success("Database loaded from .sql ✅")
    else:
        st.info("No database loaded yet.")
    # URL input for RAG with websites
    st.text_input(
        "🌐 Introduce a URL",
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url",
    )

    with st.expander(
            f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"
    ):
        st.write(
            []
            if not is_vector_db_loaded
            else [source for source in st.session_state.rag_sources]
        )

# ---------- MAIN CHAT APP ----------

model_provider = st.session_state.model.split("/")[0]  # "azure-openai"
model_name = st.session_state.model.split("/")[-1]     # "gpt-4o"

# construim LLM-ul Azure
llm_stream = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_version="2024-02-15-preview",
    model_name=model_name,  # deployment name, e.g. "gpt-4o"
    openai_api_key=AZURE_API_KEY,
    openai_api_type="azure",
    temperature=0.3,
    streaming=True,
)

# afișăm istoricul
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# mesaj nou
if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # convertim istoricul în mesaje LangChain
        lc_history = [
            HumanMessage(content=m["content"])
            if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]

        if mode == "Normal Chat":
            st.write_stream(stream_llm_response(llm_stream, lc_history))

        elif mode == "RAG":
            st.write_stream(stream_llm_rag_response(llm_stream, lc_history))

        elif mode == "DB Assistant":
            # aici folosim schema DB-ului + history
            st.write_stream(stream_db_response(llm_stream, lc_history))

