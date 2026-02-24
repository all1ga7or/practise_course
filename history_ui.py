import tkinter as tk
from tkinter import ttk
from database import get_all_runs, get_run_config

BG = "#0b1220"
CARD = "#0f172a"
TEXT = "#e5e7eb"
BORDER = "#1e293b"


def open_history_window(parent, load_callback):
    win = tk.Toplevel(parent)
    win.title("Історія запусків")
    win.configure(bg=BG)
    win.geometry("1000x500")

    frame = tk.Frame(win, bg=CARD, padx=16, pady=16)
    frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(
        frame,
        text="Історія обчислювальних експериментів",
        bg=CARD,
        fg=TEXT,
        font=("Inter", 14, "bold")
    ).pack(anchor="w", pady=(0, 10))

    columns = (
        "id", "time", "m", "pop", "gen",
        "mut", "T", "k", "best", "time_exec"
    )

    tree = ttk.Treeview(frame, columns=columns, show="headings", height=16)

    headings = {
        "id": "ID",
        "time": "Дата",
        "m": "m",
        "pop": "Популяція",
        "gen": "Покоління",
        "mut": "Мутація",
        "T": "T",
        "k": "k",
        "best": "Оптимум",
        "time_exec": "Час (с)"
    }

    for col, text in headings.items():
        tree.heading(col, text=text)
        tree.column(col, anchor="center")

    tree.pack(fill=tk.BOTH, expand=True)

    # ---------- Заповнення ----------
    for row in get_all_runs():
        tree.insert("", tk.END, values=row)

    # ---------- Подвійний клік ----------
    def on_double_click(event):
        selected = tree.selection()
        if not selected:
            return

        run_id = tree.item(selected[0], "values")[0]
        cfg = get_run_config(run_id)

        if cfg is not None:
            load_callback(cfg)

        win.destroy()

    tree.bind("<Double-1>", on_double_click)
