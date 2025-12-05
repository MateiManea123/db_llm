import os
import dotenv
from time import time
import streamlit as st

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 10

AZURE_API_KEY = os.getenv("AZ_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZ_OPENAI_ENDPOINT")

if not AZURE_API_KEY or not AZURE_ENDPOINT:
    # aici nu avem st încă (dacă e importat înainte de streamlit), deci doar logăm
    print("WARNING: AZ_OPENAI_API_KEY sau AZ_OPENAI_ENDPOINT lipsesc din .env")


# ---------- Helper: merge docs in context string ----------

def _format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ---------- LLM simple (fără RAG) ----------

def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# ---------- Indexing phase ----------

def load_doc_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(
                            f"Error loading document {doc_file.name}: {e}", icon="⚠️"
                        )
                        print(f"Error loading document {doc_file.name}: {e}")

                    finally:
                        os.remove(file_path)

                else:
                    st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(
                f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.",
                icon="✅",
            )


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)
                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(
                        f"Document from URL *{url}* loaded successfully.", icon="✅"
                    )
            else:
                st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")


def initialize_vector_db(docs):
    """Creează Chroma Vector DB folosind Azure OpenAI embeddings (text-embedding-3-large)."""
    if not AZURE_API_KEY or not AZURE_ENDPOINT:
        st.error(
            "AZ_OPENAI_API_KEY sau AZ_OPENAI_ENDPOINT nu sunt setate în .env, "
            "nu pot crea embeddings Azure."
        )
        st.stop()

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=f"{str(time()).replace('.', '')[:14]}_"
        + st.session_state["session_id"],
    )

    # limităm numărul de colecții Chroma în memorie
    chroma_client = vector_db._client
    collection_names = sorted(
        [collection.name for collection in chroma_client.list_collections()]
    )
    print("Number of collections:", len(collection_names))
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db


def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )

    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        # inițializăm vector_db cu primele documente tăiate în chunks
        st.session_state.vector_db = initialize_vector_db(document_chunks)
    else:
        st.session_state.vector_db.add_documents(document_chunks)


# ---------- RAG phase (LCEL) ----------

def get_conversational_rag_chain(llm):
    """
    Construieste un lanț RAG de tip:
      {"context", "messages", "input"} -> prompt -> llm -> text
    fără să folosească langchain.chains.*
    """
    retriever = st.session_state.vector_db.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. You will have to answer user's queries.
You will have some context to help with your answers, but it will not always be perfectly relevant.
You can also use your own knowledge to assist the user.

{context}""",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
        ]
    )

    # LCEL: construim dictul de intrare pentru prompt
    # - context: derivat din input -> retriever -> format_docs
    # - messages: istoricul conversației
    # - input: întrebarea curentă
    def _get_input(d):
        return d["input"]

    def _get_messages(d):
        return d["messages"]

    rag_chain = (
        {
            "context": RunnableLambda(_get_input)
            | retriever
            | RunnableLambda(_format_docs),
            "messages": RunnableLambda(_get_messages),
            "input": RunnableLambda(_get_input),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    payload = {"messages": messages[:-1], "input": messages[-1].content}

    for chunk in conversation_rag_chain.stream(payload):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})
