import os
import uuid
from typing import Iterable, List

import streamlit as st
from langchain_core.messages import SystemMessage
import pytds  # from python-tds package


# ----------------- Config pentru SQL Server & backup -----------------

SQLSERVER_HOST = os.getenv("SQLSERVER_HOST", "sqlserver")
SQLSERVER_PORT = int(os.getenv("SQLSERVER_PORT", "1433"))
SQLSERVER_USER = os.getenv("SQLSERVER_USER", "sa")
SQLSERVER_PASSWORD = os.getenv("SQLSERVER_PASSWORD", "ParolaTa123!")

# Folderul în care aplicația (containerul app) salvează fișierele .bak
BAK_DIR = os.getenv("BAK_DIR", "/backups")
# Folderul în care SQL Server (containerul sqlserver) vede aceleași .bak
BACKUP_SQL_DIR = os.getenv("BACKUP_SQL_DIR", "/var/opt/mssql/backups")

os.makedirs(BAK_DIR, exist_ok=True)


# ----------------- Conexiune la SQL Server -----------------


def get_sqlserver_connection(database: str = "master"):
    conn = pytds.connect(
        server=SQLSERVER_HOST,
        user=SQLSERVER_USER,
        password=SQLSERVER_PASSWORD,
        database=database,
        port=SQLSERVER_PORT,
        as_dict=True
    )
    return conn

# ----------------- Gestionare DB curentă în session_state -----------------


def has_db() -> bool:
    """
    Verifică dacă avem o bază de date SQL Server restaurată pentru sesiunea curentă.
    """
    return "current_db_name" in st.session_state and bool(
        st.session_state.current_db_name
    )


def get_db_connection():
    """
    Returnează o conexiune către baza de date SQL Server curentă.
    """
    if not has_db():
        raise RuntimeError("No DB loaded. Please upload a .bak file first.")

    db_name = st.session_state.current_db_name
    return get_sqlserver_connection(database=db_name)


# ----------------- RESTORE din .bak -----------------


def restore_database_from_bak(db_name: str, bak_sql_path: str):
    """
    Restaurează un .bak într-o bază SQL Server nouă (db_name).

      1. RESTORE FILELISTONLY ca să aflăm logical file names (pentru data & log)
      2. RESTORE DATABASE ... WITH MOVE ... în /var/opt/mssql/data

    Presupune că SQL Server rulează în containerul oficial, care folosește
    /var/opt/mssql/data ca folder pentru data files.
    """
    conn = get_sqlserver_connection(database="master")
    conn.autocommit = True
    cur = conn.cursor()

    # 1. aflăm logical file names din backup
    cur.execute(f"RESTORE FILELISTONLY FROM DISK = N'{bak_sql_path}'")
    file_rows = cur.fetchall()
    if len(file_rows) < 2:
        conn.close()
        raise RuntimeError(
            "Unexpected FILELISTONLY result; expected at least data+log files."
        )

    logical_data_name = file_rows[0]["LogicalName"]
    logical_log_name = file_rows[1]["LogicalName"]

    data_path = f"/var/opt/mssql/data/{db_name}_Data.mdf"
    log_path = f"/var/opt/mssql/data/{db_name}_Log.ldf"

    restore_sql = f"""
    RESTORE DATABASE [{db_name}]
    FROM DISK = N'{bak_sql_path}'
    WITH REPLACE,
         MOVE N'{logical_data_name}' TO N'{data_path}',
         MOVE N'{logical_log_name}' TO N'{log_path}';
    """
    cur.execute(restore_sql)
    conn.close()


# ----------------- Upload generic de DB (din .bak) -----------------


def load_db_generic():
    """
    Callback pentru file_uploader (app.py):

      - utilizatorul încarcă un fișier .bak (SQL Server backup)
      - îl salvăm într-un folder comun (BAK_DIR, montat ca volum)
      - dăm RESTORE DATABASE în SQL Server (bază nouă, cu nume unic)
      - salvăm numele DB-ului în st.session_state.current_db_name
    """
    if "db_file" not in st.session_state or not st.session_state.db_file:
        return

    file = st.session_state.db_file
    filename = file.name

    if not filename.lower().endswith(".bak"):
        st.session_state.db_loaded = False
        st.error("Unsupported file type. Please upload a .bak (SQL Server backup).")
        return

    try:
        # Salvăm .bak în containerul aplicației, în BAK_DIR
        unique_id = str(uuid.uuid4())[:8]
        bak_filename = f"{unique_id}_{filename}"
        bak_app_path = os.path.join(BAK_DIR, bak_filename)

        with open(bak_app_path, "wb") as f:
            f.write(file.read())

        # Calea văzută de SQL Server (în containerul sqlserver)
        bak_sql_path = os.path.join(BACKUP_SQL_DIR, bak_filename)

        # Numele bazei de date în SQL Server (unul per upload/sesiune)
        db_name = f"userdb_{unique_id}"

        # RESTORE DATABASE
        restore_database_from_bak(db_name, bak_sql_path)

        st.session_state.current_db_name = db_name
        st.session_state.db_loaded = True
        st.toast(
            f"✅ SQL Server DB restored from {filename} as {db_name}",
            icon="✅",
        )

    except Exception as e:
        st.session_state.db_loaded = False
        st.error(f"Error restoring DB from .bak: {e}")


# ----------------- Schema summary (pentru promptul LLM) -----------------


def get_schema_summary() -> str:
    """
    Returnează un string cu schema bazei de date SQL Server curente
    (tabele + coloane), pentru a fi inserată în promptul LLM-ului.
    """
    if not has_db():
        return "NO_DB"

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Luăm toate tabelele user (BASE TABLE)
        cur.execute(
            """
            SELECT TABLE_SCHEMA, TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_SCHEMA, TABLE_NAME;
            """
        )
        tables = cur.fetchall()

        if not tables:
            return "NO_TABLES"

        schema_lines: List[str] = []

        for row in tables:
            schema = row["TABLE_SCHEMA"]
            table = row["TABLE_NAME"]

            cur.execute(
                """
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                ORDER BY ORDINAL_POSITION;
                """,
                (schema, table),
            )
            cols = cur.fetchall()
            col_defs = ", ".join(
                f"{c['COLUMN_NAME']} {c['DATA_TYPE']}" for c in cols
            )
            schema_lines.append(f"TABLE {schema}.{table}: {col_defs}")

        return "\n".join(schema_lines)

    finally:
        conn.close()


# ----------------- LLM streaming pentru DB Assistant -----------------


def stream_db_response(llm_stream, history_messages: Iterable):
    """
    LLM care răspunde la întrebări despre baza de date SQL Server.

    Parametri:
      - llm_stream: un obiect LLM cu metoda .stream(messages)
      - history_messages: listă de mesaje LangChain (UserMessage, AIMessage etc.)

    Comportament:
      - verifică dacă există o DB încărcată din .bak
      - extrage schema cu get_schema_summary()
      - construiește un SystemMessage cu instrucțiuni clare pentru SQL Server (T-SQL)
      - transmite mesajele către LLM și face stream în UI
    """

    # 1. trebuie să avem DB
    if not has_db():
        msg = "No database loaded. Please upload a .bak (SQL Server backup) first."
        st.session_state.messages.append({"role": "assistant", "content": msg})
        yield msg
        return

    # 2. schema DB-ului
    schema_text = get_schema_summary()
    if schema_text in ("NO_DB", "NO_TABLES"):
        msg = "Database is empty or has no user tables. Please check your .bak file."
        st.session_state.messages.append({"role": "assistant", "content": msg})
        yield msg
        return

    # 3. System prompt pentru LLM
    system_content = f"""You are a SQL assistant over a Microsoft SQL Server database.

The database engine is SQL Server. Use T-SQL syntax compatible with SQL Server.

Here is the database schema (tables and columns):

{schema_text}

When the user asks something, you MUST:
1. Think step by step about what they want.
2. Propose one or more T-SQL queries that answer the question or modify the data.
3. Explain the query in natural language.
4. Return the SQL inside a fenced code block like:

```sql
SELECT ...
Do NOT invent tables or columns that are not in the schema.
If something cannot be done with this schema, clearly say so.
"""
    lc_messages = [SystemMessage(content=system_content)]
    lc_messages.extend(history_messages)

    response_message = ""

    # 4. Facem stream din LLM
    for chunk in llm_stream.stream(lc_messages):
        response_message += chunk.content
        # chunk e un obiect LangChain (AIMessageChunk); îl dăm mai departe caller-ului
        yield chunk

    # 5. persistăm răspunsul în session_state pentru UI
    st.session_state.messages.append({"role": "assistant", "content": response_message})
