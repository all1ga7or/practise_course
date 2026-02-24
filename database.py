import sqlite3
import json
from datetime import datetime
import os
import numpy as np

DB_PATH = "results/results.db"


# ================== CONNECTION ==================
def get_connection():
    return sqlite3.connect(DB_PATH)


# ================== INIT DATABASE ==================
def init_db():
    os.makedirs("results", exist_ok=True)

    conn = get_connection()
    cur = conn.cursor()

    # ---------- Запуски ----------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        dimension INTEGER,
        population INTEGER,
        generations INTEGER,
        mutation REAL,
        disturbances_T INTEGER,
        k REAL,
        best_value REAL,
        elapsed_time REAL,
        is_manual INTEGER DEFAULT 0
    )
    """)

    # ---------- Початковий стан ----------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS run_config (
        run_id INTEGER PRIMARY KEY,
        A0 TEXT,
        B0 TEXT,
        C0 TEXT,
        u_min TEXT,
        u_max TEXT,
        FOREIGN KEY(run_id) REFERENCES runs(id)
    )
    """)

    # ---------- Динаміка у часі ----------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS evolution (
        run_id INTEGER,
        t INTEGER,
        fitness REAL,
        utility REAL,
        A TEXT,
        B TEXT,
        C TEXT,
        u TEXT,
        alpha TEXT,
        beta TEXT,
        gamma TEXT,
        elapsed_time REAL,
        effect_percent REAL,
        FOREIGN KEY(run_id) REFERENCES runs(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS scenarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        description TEXT,
        m INTEGER,
        alpha TEXT, 
        beta TEXT, 
        gamma TEXT  
    )
    """)

    cur.execute("SELECT COUNT(*) FROM scenarios")
    if cur.fetchone()[0] == 0:
        sample_alpha = [1.12, 1.05, 1.02, 1.20]
        sample_beta = [0.95, 1.10, 1.00, 0.80]
        sample_gamma = [1.05, 0.98, 1.03, 1.10]
        
        cur.execute("""
        INSERT INTO scenarios (name, description, m, alpha, beta, gamma)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "Енерго-трансформація 2024",
            "Модель збурень в енергетиці: ріст цін на паливо, підтримка ВДЕ та еко-податки",
            4,
            json.dumps(sample_alpha),
            json.dumps(sample_beta),
            json.dumps(sample_gamma)
        ))

    conn.commit()
    conn.close()


# ================== SAVE RUN ==================
def save_run(params, best_value=None, elapsed_time=None, is_manual=0):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO runs (
        timestamp, dimension, population,
        generations, mutation, disturbances_T, k,
        best_value, elapsed_time, is_manual
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        params["dimension"],
        params["population"],
        params["generations"],
        params["mutation"],
        params["disturbances"],
        params["k"],
        best_value,
        elapsed_time,
        is_manual
    ))

    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


# ================== SAVE INITIAL CONFIG ==================
def save_run_config(run_id, A0, B0, C0, u_min, u_max):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT OR REPLACE INTO run_config
    (run_id, A0, B0, C0, u_min, u_max)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        json.dumps(A0.tolist()),
        json.dumps(B0.tolist()),
        json.dumps(C0.tolist()),
        json.dumps(u_min.tolist()),
        json.dumps(u_max.tolist())
    ))

    conn.commit()
    conn.close()


# ================== SAVE TIME STEP ==================
def save_time_step(run_id, t, fitness, utility, A, B, C, u, elapsed_time, alpha=None, beta=None, gamma=None, effect_percent=None):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO evolution
    (run_id, t, fitness, utility, A, B, C, u, alpha, beta, gamma, elapsed_time, effect_percent)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        t,
        float(fitness),
        float(utility),
        json.dumps(A.tolist()),
        json.dumps(B.tolist()),
        json.dumps(C.tolist()),
        json.dumps(u.tolist()),
        json.dumps(alpha.tolist()),
        json.dumps(beta.tolist()),
        json.dumps(gamma.tolist()),
        elapsed_time,
        float(effect_percent) if effect_percent is not None else 0.0
    ))

    conn.commit()
    conn.close()


# ================== UPDATE RUN ==================
def update_run(run_id, best_value, elapsed_time):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    UPDATE runs
    SET best_value = ?, elapsed_time = ?
    WHERE id = ?
    """, (float(best_value), elapsed_time, run_id))

    conn.commit()
    conn.close()


# ================== GET ALL RUNS ==================
def get_all_runs():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT id, timestamp, dimension, population,
           generations, mutation, disturbances_T, k,
           best_value, elapsed_time
    FROM runs
    ORDER BY id DESC
    """)

    rows = cur.fetchall()
    conn.close()
    return rows


# ================== LOAD FULL CONFIG ==================
def get_run_config(run_id):
    conn = get_connection()
    cur = conn.cursor()

    # ---- runs ----
    cur.execute("""
    SELECT dimension, population, generations,
           mutation, disturbances_T, k
    FROM runs
    WHERE id = ?
    """, (run_id,))
    run = cur.fetchone()

    # ---- initial config ----
    cur.execute("""
    SELECT A0, B0, C0, u_min, u_max
    FROM run_config
    WHERE run_id = ?
    """, (run_id,))
    cfg = cur.fetchone()

    conn.close()

    if run is None or cfg is None:
        return None

    (dimension, population, generations,
     mutation, disturbances_T, k) = run

    A0, B0, C0, u_min, u_max = cfg

    return {
        "run_id": run_id,
        "dimension": dimension,
        "population": population,
        "generations": generations,
        "mutation": mutation,
        "disturbances": disturbances_T,
        "k": k,
        "A": np.array(json.loads(A0)),
        "B": np.array(json.loads(B0)),
        "C": np.array(json.loads(C0)),
        "u_min": np.array(json.loads(u_min)),
        "u_max": np.array(json.loads(u_max)),
    }

# ================== GET EVOLUTION ==================
def get_evolution(run_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT t, fitness, utility, A, B, C, u, alpha, beta, gamma, elapsed_time, effect_percent
    FROM evolution
    WHERE run_id = ?
    ORDER BY t
    """, (run_id,))

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "t": r[0],
            "fitness": r[1],
            "utility": r[2],
            "A": np.array(json.loads(r[3])),
            "B": np.array(json.loads(r[4])),
            "C": np.array(json.loads(r[5])),
            "u": np.array(json.loads(r[6])),
            "alpha": np.array(json.loads(r[7])),
            "beta": np.array(json.loads(r[8])),
            "gamma": np.array(json.loads(r[9])),
            "elapsed_time": r[10],
            "effect_percent": r[11]
        }
        for r in rows
    ]

# ================== SCENARIOS ==================
def get_all_scenarios():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name, description, m, alpha, beta, gamma FROM scenarios")
    rows = cur.fetchall()
    conn.close()
    return rows

def save_new_scenario(name, desc, m, a, b, g):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO scenarios (name, description, m, alpha, beta, gamma)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (name, desc, m, json.dumps(a), json.dumps(b), json.dumps(g)))
    conn.commit()
    conn.close()