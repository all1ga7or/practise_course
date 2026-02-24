import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import csv

BG = "#0f172a"
CARD = "#111827"
TEXT = "#e5e7eb"
BORDER = "#1f2937"
FONT_UI = ("Inter", 10)

def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

def create_matrix(frame, size, label):
    clear_frame(frame)

    tk.Label(
        frame, text=label,
        bg=CARD, fg=TEXT,
        font=("Inter", 11, "bold")
    ).pack(anchor="w", pady=(0, 6))

    grid = tk.Frame(frame, bg=CARD)
    grid.pack()

    entries = []
    for i in range(size):
        row = []
        for j in range(size):
            e = tk.Entry(
                grid, width=6, justify="center",
                bg=BG, fg=TEXT,
                insertbackground=TEXT,
                relief="flat", highlightthickness=1,
                highlightbackground=BORDER,
                font=FONT_UI
            )
            e.grid(row=i, column=j, padx=4, pady=4)
            row.append(e)
        entries.append(row)
    return entries

def create_vector(frame, size, label):
    clear_frame(frame)

    tk.Label(
        frame, text=label,
        bg=CARD, fg=TEXT,
        font=("Inter", 11, "bold")
    ).pack(anchor="w", pady=(0, 6))

    grid = tk.Frame(frame, bg=CARD)
    grid.pack()

    entries = []
    for i in range(size):
        e = tk.Entry(
            grid, width=6, justify="center",
            bg=BG, fg=TEXT,
            insertbackground=TEXT,
            relief="flat", highlightthickness=1,
            highlightbackground=BORDER,
            font=FONT_UI
        )
        e.grid(row=i, column=0, padx=4, pady=4)
        entries.append(e)
    return entries

def autofill(entries, low=0, high=1):
    for row in entries:
        if isinstance(row, list):
            for e in row:
                e.delete(0, tk.END)
                e.insert(0, f"{np.random.uniform(low, high):.2f}")
        else:
            row.delete(0, tk.END)
            row.insert(0, f"{np.random.uniform(low, high):.2f}")

def read_matrix(entries):
    return np.array([
        [float(e.get()) if e.get() else 0.0 for e in row]
        for row in entries
    ])

def read_vector(entries):
    return np.array([
        float(e.get()) if e.get() else 0.0
        for e in entries
    ])

def load_abc_from_csv(m, a_entries, b_entries, c_entries):
    # 1. Відкриваємо вікно файлової системи
    filepath = filedialog.askopenfilename(
        title="Оберіть файл з початковими даними (A, B, C)",
        filetypes=[("CSV файли", "*.csv"), ("Текстові файли", "*.txt"), ("Усі файли", "*.*")]
    )
    
    # Якщо користувач натиснув "Скасувати", виходимо
    if not filepath:
        return

    try:
        with open(filepath, newline='', encoding='utf-8') as f:
            # Використовуємо ; або , як роздільник
            content = f.read()
            dialect = csv.Sniffer().sniff(content[:1024])
            f.seek(0)
            reader = list(csv.reader(f, dialect))
            
            # Перевірка чи достатньо рядків (m для A + 1 для B + 1 для C)
            if len(reader) < m + 2:
                raise ValueError(f"Файл має містити мінімум {m+2} рядків")

            # 2. Заповнюємо матрицю A (рядки від 0 до m-1)
            for i in range(m):
                for j in range(m):
                    a_entries[i][j].delete(0, tk.END)
                    a_entries[i][j].insert(0, reader[i][j].strip())
            
            # 3. Заповнюємо вектор B (рядок m)
            for j in range(m):
                b_entries[j].delete(0, tk.END)
                b_entries[j].insert(0, reader[m][j].strip())
                
            # 4. Заповнюємо вектор C (рядок m + 1)
            for j in range(m):
                c_entries[j].delete(0, tk.END)
                c_entries[j].insert(0, reader[m + 1][j].strip())
                
        messagebox.showinfo("Успіх", "Дані ABC успішно імпортовано!")
        
    except Exception as e:
        messagebox.showerror("Помилка імпорту", f"Не вдалося зчитати файл.\nПеревірте формат (має бути {m} рядків для A, потім B та C).\nДеталі: {e}")
