import os
import dotenv
import streamlit as st

from db_methods import stream_db_response, has_db, load_db_generic

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# ----------------- Config & helpers -----------------

dotenv.load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT = (
    os.getenv("AZURE_OPENAI_DEPLOYMENT")
    or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    or "gpt-4o"  # schimbă dacă ai alt nume de deployment
)

st.set_page_config(
    page_title="Blue SQL Copilot",
    page_icon="💙",
    layout="wide",
)

# Custom CSS pentru tematica albastră
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 45%, #020617 100%);
        color: #e5e7eb;
    }
    /* Container central mai îngust pentru chat */
    .main > div {
        max-width: 1100px;
        margin: 0 auto;
    }
    /* Titlu mare */
    .blue-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #60a5fa);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 0.25rem;
    }
    .blue-subtitle {
        font-size: 0.98rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    /* Card de info */
    .info-card {
        border-radius: 16px;
        padding: 1rem 1.25rem;
        border: 1px solid rgba(59,130,246,0.4);
        background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(15,23,42,0.6));
        margin-bottom: 1.5rem;
    }
    .info-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.95rem;
        color: #bfdbfe;
    }
    .info-card p {
        margin: 0;
        font-size: 0.9rem;
        color: #e5e7eb;
    }
    /* Chat bubbles */
    .stChatMessage[data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(148,163,184,0.4);
        background: rgba(15,23,42,0.85);
    }
    .stChatMessage[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"] strong) {
        border-color: rgba(59,130,246,0.6);
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #020617);
        border-right: 1px solid rgba(30,64,175,0.7);
    }
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e5e7eb;
        margin-bottom: 0.25rem;
    }
    .sidebar-subtitle {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 1rem;
    }
    .sidebar-section-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #bfdbfe;
        margin-top: 1.2rem;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Session state -----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------- Construim LLM-ul Azure (doar unul, fără selecție de model) -----------------

if not AZURE_ENDPOINT or not AZURE_API_KEY:
    llm_stream = None
else:
    llm_stream = AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        openai_api_version=AZURE_API_VERSION,
        model_name=AZURE_DEPLOYMENT,
        openai_api_key=AZURE_API_KEY,
        openai_api_type="azure",
        temperature=0.2,
        streaming=True,
    )

# ----------------- SIDEBAR: Setup DB -----------------

with st.sidebar:
    st.markdown('<div class="sidebar-title">💙 Blue SQL Copilot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-subtitle">Asistent specializat doar pe baza ta de date SQL Server.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">1️⃣ Încarcă backup-ul (.bak)</div>', unsafe_allow_html=True)

    st.file_uploader(
        "Upload SQL Server .bak",
        type=["bak"],
        key="db_file",
        on_change=load_db_generic,
    )

    st.markdown('<div class="sidebar-section-title">2️⃣ Status bază de date</div>', unsafe_allow_html=True)

    if has_db():
        st.success("Baza de date este încărcată și gata de interogat. ✅")
    else:
        st.info("Încarcă un fișier .bak pentru a porni asistentul. ℹ️")

    st.markdown("---")
    st.caption(
        "Asistentul generează T-SQL pe baza schemei bazei tale de date și explică rezultatul în limbaj natural."
    )

# ----------------- MAIN: Chat doar pentru DB -----------------

st.markdown('<div class="blue-title">Blue SQL Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="blue-subtitle">Pune întrebări în română despre baza ta de date SQL Server. '
    "Asistentul gândește interogări T-SQL și le explică.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">
      <h4>ℹ️ Cum îl poți folosi</h4>
      <p>
        Întreabă lucruri de genul:<br/>
        • „Câți angajați activi sunt în baza de date?”<br/>
        • „Arată-mi top 5 produse după vânzări.”<br/>
        • „Există comenzi fără client asociat?”<br/>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not llm_stream:
    st.error("Lipsesc variabilele de mediu pentru Azure OpenAI (endpoint / key).")
    st.stop()

if not has_db():
    st.warning("Încarcă un fișier .bak în sidebar pentru a începe conversația.")
    # Totuși, afișăm istoricul (dacă există), dar nu permitem input nou
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.stop()

# Afișăm istoricul de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input nou de la utilizator (doar chat DB)
prompt = st.chat_input("Scrie o întrebare despre baza de date...")

if prompt:
    # salvăm mesajul utilizatorului
    st.session_state.messages.append({"role": "user", "content": prompt})

    # randăm mesajul utilizatorului
    with st.chat_message("user"):
        st.markdown(prompt)

    # pregătim istoricul pentru LLM (HumanMessage / AIMessage)
    lc_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in st.session_state.messages
    ]

    # mesajul asistentului (DB Assistant only)
    with st.chat_message("assistant"):
        st.write_stream(stream_db_response(llm_stream, lc_history))
