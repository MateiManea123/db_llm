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


def stream_db_response(llm, history_messages):
    """
    DB Assistant "inteligent":
      - Dacă întrebarea poate fi rezolvată cu SQL:
          1) generează T-SQL
          2) execută query-ul pe SQL Server
          3) explică rezultatul în limbaj natural
      - Dacă nu se poate (SQL invalid / întrebare conceptuală):
          -> răspunde ca un LLM normal, folosind schema DB ca context.
    """

    # --- 0. verificăm DB ---
    if not has_db():
        msg = "Nu există nicio bază de date încărcată. Încarcă un fișier .bak mai întâi."
        st.session_state.messages.append({"role": "assistant", "content": msg})
        yield msg
        return

    schema_text = get_schema_summary()
    if schema_text in ("NO_DB", "NO_TABLES"):
        msg = "Baza de date este goală sau nu are tabele de utilizator. Verifică fișierul .bak."
        st.session_state.messages.append({"role": "assistant", "content": msg})
        yield msg
        return

    # ultima întrebare a user-ului
    user_question = None
    if history_messages:
        last_msg = history_messages[-1]
        try:
            user_question = last_msg.content
        except AttributeError:
            user_question = str(last_msg)
    if not user_question:
        msg = "Nu am putut determina întrebarea utilizatorului."
        st.session_state.messages.append({"role": "assistant", "content": msg})
        yield msg
        return

    # ---- helper pentru fallback: răspuns ca LLM normal peste schema DB ----

    def fallback_normal_answer(reason: str = ""):
        system_fallback = (
            "You are an AI assistant that answers questions about a Microsoft SQL Server database.\n\n"
            "You have access to the database schema (tables and columns) below.\n"
            "Use it as context, but you do NOT need to always write SQL.\n"
            "You can answer conceptually, explain relationships, suggest queries, etc.\n\n"
            "Database schema:\n\n"
            f"{schema_text}\n\n"
            "Answer the user's question in a clear and friendly way (in Romanian if the user writes in Romanian)."
        )

        messages = [SystemMessage(content=system_fallback)]
        messages.extend(history_messages)

        response = llm.invoke(messages)
        answer = response.content or "Nu am reușit să formulez un răspuns."

        if reason:
            # opțional: poți comenta motivul în debug log, dar nu e obligatoriu să îl afișezi user-ului
            print(f"[DB ASSISTANT FALLBACK] {reason}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        return answer

    # --- 1. ÎNCERCĂM varianta cu SQL ---

    # 1.1. Generăm SQL cu un prompt specializat
    system_generate_sql = (
        "You are an assistant that generates ONLY T-SQL queries for Microsoft SQL Server.\n\n"
        "The database engine is SQL Server. Use standard T-SQL.\n\n"
        "You will receive the user's last question and the database schema.\n"
        "Your task:\n"
        "- Understand the question.\n"
        "- Generate EXACTLY ONE T-SQL query that answers it.\n"
        "- Return the query inside a single fenced code block.\n"
        "- Do NOT write explanations or any other text outside the code block.\n\n"
        "Here is the database schema:\n\n"
        f"{schema_text}\n\n"
        "Return ONLY the SQL query."
    )

    sql_messages = [
        SystemMessage(content=system_generate_sql),
        history_messages[-1],  # doar ultima întrebare
    ]

    try:
        sql_response = llm.invoke(sql_messages)
    except Exception as e:
        # dacă modelul nu răspunde, facem fallback direct
        ans = fallback_normal_answer(reason=f"LLM SQL generation error: {e}")
        yield ans
        return

    raw = sql_response.content or ""

    # 1.2. Extragem SQL-ul din blocul ```...```
    sql = None
    if "```" in raw:
        first = raw.find("```") + 3
        last = raw.rfind("```")
        sql = raw[first:last].strip()
    else:
        sql = raw.strip()

    # curățăm prefixe de tip "sql"/"tsql"
    if sql:
        lines = sql.splitlines()
        if lines and lines[0].strip().lower() in ("sql", "tsql"):
            sql = "\n".join(lines[1:]).strip()

    # dacă nu a putut genera un SQL clar -> fallback
    if not sql:
        ans = fallback_normal_answer(reason="No SQL extracted from model output.")
        yield ans
        return

    # 1.3. Executăm SQL-ul
    try:
        columns, rows = run_sql_query(sql)
    except Exception as e:
        # de exemplu dacă întrebare nu are sens ca SQL
        ans = fallback_normal_answer(reason=f"SQL execution failed: {e}")
        yield ans
        return

    # --- 2. Dacă totul a mers, cerem LLM-ului să explice rezultatul ---

    max_rows_for_llm = 50
    sample_rows = rows[:max_rows_for_llm]

    result_payload = {
        "question": user_question,
        "sql": sql,
        "columns": columns,
        "rows_sample": sample_rows,
        "total_rows": len(rows),
    }

    system_explain = (
        "You are an assistant that answers questions about a Microsoft SQL Server database.\n\n"
        "You will be given:\n"
        "- the original user question (in Romanian or English),\n"
        "- the T-SQL query that was executed,\n"
        "- the query result (column names and rows, possibly truncated),\n"
        "- the total number of rows.\n\n"
        "Your job is to:\n"
        "- Explain the answer in a friendly, concise way in Romanian.\n"
        "- Use the actual query results to answer (do NOT hallucinate values).\n"
        "- If the result is a single value (1 row, 1 column), highlight that value.\n"
        "- Optionally, show a small Markdown table if there are multiple rows.\n"
        "- You may optionally show the SQL query at the end in a code block.\n"
    )

    explain_messages = [
        SystemMessage(content=system_explain),
        SystemMessage(content=f"Execution context (Python-style dict):\n\n{result_payload}"),
    ]

    explain_response = llm.invoke(explain_messages)
    final_answer = explain_response.content or "Nu am reușit să formulez un răspuns."

    st.session_state.messages.append(
        {"role": "assistant", "content": final_answer}
    )

    yield final_answer




# --------------------------------------------------------------
# RUN SQL QUERY (folosit de stream_db_response)
# --------------------------------------------------------------

def run_sql_query(sql: str):
    """
    Rulează un query T-SQL pe baza curentă.
    Returnează: (lista_de_coloane, lista_de_rânduri)
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()

        # extragem numele de coloane din cursor.description
        if cur.description:
            columns = [col[0] for col in cur.description]
        else:
            columns = []

        # dacă avem dict-uri din pytds
        if rows and isinstance(rows[0], dict):
            normalized = [[r.get(col) for col in columns] for r in rows]
        else:
            normalized = [list(r) for r in rows]

        return columns, normalized
    finally:
        conn.close()
