import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from database import get_all_scenarios, save_new_scenario

def open_scenario_selection(parent, current_m):
    win = tk.Toplevel(parent)
    win.title("Менеджер сценаріїв")
    win.geometry("1000x700")
    win.configure(bg="#0b1220")
    
    selected_data = {"result": None}
    all_scenarios_list = []

    notebook = ttk.Notebook(win)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # --- ВКЛАДКА 1: ВИБІР І ВІЗУАЛІЗАЦІЯ ---
    view_tab = tk.Frame(notebook, bg="#0f172a")
    notebook.add(view_tab, text=" Мої сценарії ")

    paned = tk.PanedWindow(view_tab, orient=tk.HORIZONTAL, bg="#1e293b", sashwidth=4)
    paned.pack(fill=tk.BOTH, expand=True)

    left_frame = tk.Frame(paned, bg="#0f172a")
    right_frame = tk.Frame(paned, bg="#0f172a")
    paned.add(left_frame, width=400)
    paned.add(right_frame)

    columns = ("name", "m")
    tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=15)
    tree.heading("name", text="Назва сценарію")
    tree.heading("m", text="m")
    tree.column("m", width=50, anchor="center")
    tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    desc_label = tk.Label(left_frame, text="Опис: ", fg="#94a3b8", bg="#0f172a", wraplength=350, justify="left")
    desc_label.pack(fill=tk.X, padx=10, pady=5)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_preview(event):
        item = tree.selection()
        if not item: return
        name = tree.item(item, "values")[0]
        for s in all_scenarios_list:
            if s[0] == name:
                desc_label.config(text=f"Опис: {s[1]}")
                alpha = json.loads(s[3]); beta = json.loads(s[4]); gamma = json.loads(s[5])
                ax.clear()
                x = np.arange(len(alpha))
                width = 0.25
                ax.bar(x - width, alpha, width, label='Alpha (C)', color='#38bdf8')
                ax.bar(x, beta, width, label='Beta (B)', color='#10b981')
                ax.bar(x + width, gamma, width, label='Gamma (A)', color='#fbbf24')
                ax.axhline(1.0, color='white', linestyle='--', alpha=0.3)
                ax.set_title(f"Коефіцієнти збурення: {name}", color="white")
                ax.set_xticks(x); ax.set_xticklabels([f"S{i+1}" for i in x])
                ax.tick_params(colors='white'); ax.legend()
                fig.tight_layout(); canvas.draw()
                break

    tree.bind("<<TreeviewSelect>>", update_preview)

    def refresh_tree():
        nonlocal all_scenarios_list
        for item in tree.get_children(): tree.delete(item)
        all_scenarios_list = get_all_scenarios()
        for s in all_scenarios_list:
            if s[2] == current_m:
                tree.insert("", tk.END, values=(s[0], s[2]))

    refresh_tree()

    def apply_selection():
        item = tree.selection()
        if not item: return
        name = tree.item(item, "values")[0]
        for s in all_scenarios_list:
            if s[0] == name:
                selected_data["result"] = {
                    "alpha": json.loads(s[3]),
                    "beta": json.loads(s[4]),
                    "gamma": json.loads(s[5])
                }
                win.destroy()
                break

    tk.Button(left_frame, text="✅ Застосувати сценарій", bg="#38bdf8", fg="black", 
              font=("Inter", 10, "bold"), command=apply_selection, pady=10).pack(fill=tk.X, padx=10, pady=10)
    
    def reset_to_random():
        selected_data["result"] = None 
        win.destroy()

    tk.Button(left_frame, text="🎲 Випадкові збурення (скинути)", bg="#475569", fg="white", 
              font=("Inter", 10), command=reset_to_random, pady=8).pack(fill=tk.X, padx=10, pady=5)

    # --- ВКЛАДКА 2: СТВОРЕННЯ/ІМПОРТ ---
    create_tab = tk.Frame(notebook, bg="#0f172a")
    notebook.add(create_tab, text=" Додати новий ")

    meta_frame = tk.Frame(create_tab, bg="#0f172a")
    meta_frame.pack(fill=tk.X, padx=20, pady=10)

    tk.Label(meta_frame, text="Назва:", fg="white", bg="#0f172a").grid(row=0, column=0, sticky="w")
    name_entry = tk.Entry(meta_frame, width=50, bg="#1e293b", fg="white", insertbackground="white")
    name_entry.grid(row=0, column=1, pady=5, padx=10)

    tk.Label(meta_frame, text="Опис:", fg="white", bg="#0f172a").grid(row=1, column=0, sticky="w")
    desc_entry = tk.Entry(meta_frame, width=50, bg="#1e293b", fg="white", insertbackground="white")
    desc_entry.grid(row=1, column=1, pady=5, padx=10)

    table_frame = tk.Frame(create_tab, bg="#0f172a")
    table_frame.pack(pady=10)

    # Заголовки таблиці
    headers = ["Сектор", "Alpha (C)", "Beta (B)", "Gamma (A)"]
    for j, h in enumerate(headers):
        tk.Label(table_frame, text=h, fg="#94a3b8", bg="#0f172a", font=("Inter", 9, "bold")).grid(row=0, column=j, padx=5)

    manual_entries = []
    for i in range(current_m):
        tk.Label(table_frame, text=f"S{i+1}", fg="white", bg="#0f172a").grid(row=i+1, column=0)
        row_e = []
        for j in range(3):
            e = tk.Entry(table_frame, width=12, justify="center", bg="#1e293b", fg="white", insertbackground="white")
            e.insert(0, "1.0")
            e.grid(row=i+1, column=j+1, padx=5, pady=2)
            row_e.append(e)
        manual_entries.append(row_e)

    # --- ЛОГІКА ІМПОРТУ CSV ---
    def load_csv_to_manual():
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                f.seek(0)
                
                # Спроба отримати метадані з назви файлу, якщо всередині немає
                filename = path.split('/')[-1].replace('.csv', '').replace('_', ' ').title()
                name_entry.delete(0, tk.END); name_entry.insert(0, filename)
                
                reader = csv.DictReader(filter(lambda row: row[0]!='#', lines))
                for i, row in enumerate(reader):
                    if i < current_m:
                        manual_entries[i][0].delete(0, tk.END); manual_entries[i][0].insert(0, row['alpha'])
                        manual_entries[i][1].delete(0, tk.END); manual_entries[i][1].insert(0, row['beta'])
                        manual_entries[i][2].delete(0, tk.END); manual_entries[i][2].insert(0, row['gamma'])
            
            messagebox.showinfo("Імпорт", f"Дані з '{filename}' успішно завантажені!")
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося зчитати CSV. Формат має бути:\nsector,alpha,beta,gamma\n\nДеталі: {e}")

    def save_to_db():
        name = name_entry.get(); desc = desc_entry.get()
        if not name: 
            messagebox.showerror("Помилка", "Введіть назву сценарію")
            return
        try:
            a = [float(row[0].get()) for row in manual_entries]
            b = [float(row[1].get()) for row in manual_entries]
            g = [float(row[2].get()) for row in manual_entries]
            save_new_scenario(name, desc, current_m, a, b, g)
            messagebox.showinfo("Успіх", f"Сценарій '{name}' збережено!")
            refresh_tree(); notebook.select(0)
        except ValueError:
            messagebox.showerror("Помилка", "Перевірте коректність чисел у таблиці")

    btn_frame = tk.Frame(create_tab, bg="#0f172a")
    btn_frame.pack(pady=20)

    tk.Button(btn_frame, text="📁 Імпорт CSV", command=load_csv_to_manual, 
              bg="#475569", fg="white", font=("Inter", 10), padx=20).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="💾 Зберегти в базу даних", command=save_to_db, 
              bg="#10b981", fg="white", font=("Inter", 10, "bold"), padx=20).pack(side=tk.LEFT, padx=10)

    win.grab_set()
    parent.wait_window(win)
    return selected_data["result"]