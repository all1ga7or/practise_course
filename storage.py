import csv
import os
from datetime import datetime

# ================== PATHS ==================
FINAL_LOG = "results/log/log.csv"      # підсумки запусків
TIME_LOG = "results/time/time.csv"     # еволюція по часу t
ABC_LOG = "results/abc/abc.csv"        # A(t), B(t), C(t)

# ================== INIT ==================
def init_storage():
    os.makedirs("results/log", exist_ok=True)
    os.makedirs("results/time", exist_ok=True)
    os.makedirs("results/abc", exist_ok=True)

    # ---- Підсумковий лог запусків ----
    if not os.path.exists(FINAL_LOG):
        with open(FINAL_LOG, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp",
                "dimension",
                "population",
                "generations",
                "mutation",
                "disturbances_T",
                "best_fitness",
                "time_sec"
            ])

    # ---- Еволюція по часу t ----
    with open(TIME_LOG, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "t",
            "fitness"
        ])

    # ---- Динаміка A(t), B(t), C(t) ----
    with open(ABC_LOG, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "t",
            "A_flat",
            "B",
            "C",
            "fitness"
        ])

# ================== LOGGING ==================
def log_time_step(t, fitness):
    """Збереження значення цільової функції в момент часу t"""
    with open(TIME_LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            t,
            round(float(fitness), 6)
        ])

def log_abc(t, A, B, C, fitness):
    """
    Збереження матриці A(t), векторів B(t), C(t)
    Матриця A зберігається як плоский список
    """
    with open(ABC_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            t,
            [round(a, 6) for row in A for a in row],
            [round(b, 6) for b in B],
            [round(c, 6) for c in C],
            round(float(fitness), 6)
        ])

# ================== FINAL RESULT ==================
def log_final(params, best_fitness, elapsed_time):
    """
    Збереження підсумку запуску алгоритму
    """
    with open(FINAL_LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            params["dimension"],
            params["population"],
            params["generations"],
            params["mutation"],
            params["disturbances"],
            round(float(best_fitness), 6),
            f"{elapsed_time:.2f}"
        ])
