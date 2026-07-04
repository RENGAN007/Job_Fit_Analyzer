import sqlite3, datetime

DB_PATH = "job_fit_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            resume_filename TEXT,
            job_snippet     TEXT,
            ats_score       INTEGER,
            output_text     TEXT
        )
    """)
    conn.commit(); conn.close()

def save_analysis(resume_filename, job_description, ats_score, output_text):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO analyses (timestamp,resume_filename,job_snippet,ats_score,output_text) VALUES (?,?,?,?,?)",
        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
         resume_filename or "Unknown",
         job_description[:120].strip() + ("..." if len(job_description) > 120 else ""),
         ats_score, output_text)
    )
    conn.commit(); conn.close()

def load_all_analyses():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM analyses ORDER BY id DESC")]
    conn.close(); return rows

def delete_analysis(row_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM analyses WHERE id=?", (row_id,))
    conn.commit(); conn.close()

def clear_all_analyses():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM analyses")
    conn.commit(); conn.close()