import random
import tkinter as tk
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from model import PenalizedModel
from genetic_algorithm import GeneticAlgorithm
from matrix_ui import *
from visualization import *
from storage import *
from dataclasses import dataclass
from tkinter import ttk
from database import (
    init_db,
    save_run,
    save_run_config,
    update_run,
    save_time_step,
    get_evolution
)
from history_ui import open_history_window
from scenario_ui import open_scenario_selection
from results_surface import build_solution_surface

# ================== INIT ==================
init_db()

# ================== THEME ==================
BG = "#0b1220"
BR = "#741ace"
CARD = "#0f172a"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"
BORDER = "#1e293b"

FONT_UI = ("Inter", 10)
FONT_TITLE = ("Inter", 14, "bold")
FONT_MONO = ("JetBrains Mono", 10)

# ================== GLOBALS ==================
utility_history = []
non_adapted_utility_history = []
fitness_history = []
replay_data = []
ui_ready = False
replay_mode = False
replay_button = None
current_loaded_cfg = None
manual_scenario_data = None

# ================== MATPLOTLIB ==================
plt.rcParams.update({
    "figure.facecolor": CARD,
    "axes.facecolor": CARD,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": BORDER,
    "text.color": TEXT
})

# ================== HELPERS ==================
def card(parent, title):
    frame = tk.Frame(parent, bg=CARD, padx=16, pady=14)
    tk.Label(frame, text=title, bg=CARD, fg=TEXT, font=FONT_TITLE)\
        .pack(anchor="w", pady=(0, 10))
    return frame

def log_info(text):
    log_panel.insert(tk.END, text + "\n")
    log_panel.see(tk.END)

def rebuild_matrices(*args):
    if not ui_ready:
        return
    global A_entries, B_entries, C_entries
    dim = dimension_var.get()
    A_entries = create_matrix(A_frame, dim, "Матриця A")
    B_entries = create_vector(B_frame, dim, "Вектор B")
    C_entries = create_vector(C_frame, dim, "Вектор C")

# ================== START ==================
def start():
    #log_panel.delete("1.0", tk.END)
    init_storage()
    fitness_history.clear()
    utility_history.clear()
    non_adapted_utility_history.clear()
    A_history, B_history, C_history = [], [], []
    global manual_scenario_data
    replay_button = None
    last_solution = None

    # ----- PARAMETERS -----
    m = dimension_var.get()
    T = disturbances_var.get()
    k = disturbance_k_var.get()

    pop = population_var.get()
    gens = generations_var.get()
    mut = mutation_var.get()

    # ----- INITIAL DATA -----
    A = read_matrix(A_entries)
    u0 = np.ones(m)
    C = read_vector(C_entries)
    C_scale = np.sum(C)
    B = read_vector(B_entries)
    B_scale = np.sum(B)
    B_norm = B / B_scale
    C_norm = C / C_scale
    
    u_min = np.full(m, 1 - k)
    u_max = np.full(m, 1 + k)
    gamma_min, gamma_max = 1 - k, 1 + k
    alpha_min, alpha_max = 1 - k, 1 + k
    beta_min,  beta_max  = 1 - k, 1 + k

    params = {
        "dimension": m,
        "population": pop,
        "generations": gens,
        "mutation": mut,
        "disturbances": T,
        "k": k
    }


    log_info("▶ Адаптивний процес запущено")
    log_info(f"Кількість збурень T = {T}, параметр збурення k = {k:.2f}")

    is_manual = 1 if manual_scenario_data else 0
    run_id = save_run(params, None, None, is_manual=is_manual)
    save_run_config(
        run_id,
        A0=A,
        B0=B,
        C0=C,
        u_min=u_min,
        u_max=u_max
    )

    # ================== TIME LOOP ==================
    for t in range(T):
        start_time = time.time()
        log_info(f"————————————— t = {t + 1} —————————————————")
        model = PenalizedModel(
            A=A,
            B=B_norm,
            C=C_norm,
            u_min=u_min,
            u_max=u_max,
            k=k
        )

        ga = GeneticAlgorithm(
            model=model,
            pop_size=pop,
            generations=gens,
            mutation_rate=mut
        )

        best_fitness, (A_opt, u_opt) = ga.run(last_best_solution=last_solution)
        best_fitness*=-1
        utility = float(np.dot(B_norm, u_opt)) * B_scale
        non_adapted_utility = float(np.dot(B_norm, u0)) * B_scale

        fitness_history.append(best_fitness)
        utility_history.append(utility)
        non_adapted_utility_history.append(non_adapted_utility)
        last_solution = np.concatenate([A_opt.flatten(), u_opt])

        elapsed = time.time() - start_time

        # ----- SYSTEM UPDATE -----
        if manual_scenario_data:
            alpha = np.array(manual_scenario_data['alpha'])
            beta  = np.array(manual_scenario_data['beta'])
            gamma = np.array(manual_scenario_data['gamma'])
        else:
            gamma = np.random.uniform(gamma_min, gamma_max, size=A.shape[1])
            alpha = np.random.uniform(alpha_min, alpha_max, size=C.shape)
            beta  = np.random.uniform(beta_min,  beta_max,  size=B.shape)

        A = A * gamma[np.newaxis, :]
        B_norm = B_norm * beta
        C_norm = C_norm * alpha

        # ================== FINISH ==================
        eco_effect = utility - non_adapted_utility
        eco_effect_percent = (eco_effect / non_adapted_utility) * 100 if non_adapted_utility != 0 else 0

        log_info(f"Найкраще значення цільової функції: {fitness_history[-1]:.4f}")
        log_info(f"Найкраще значення пристосованості: {utility:.2f}")
        log_info(f"Оптимальне u*: {u_opt}")
        log_info(f"Оптимальна структура A*:\n{A}")
        log_info(f"Час виконання: {elapsed:.2f} с")
        log_info(f"📊 Економічний ефект адаптації: {eco_effect:.3f} ({eco_effect_percent:.1f}%)")
        log_info("■ Алгоритм завершено\n")

        update_run(run_id, fitness_history[-1], elapsed)

        # ----- CSV + DB -----
        save_time_step(run_id, t, fitness_history[-1], utility, A, B_norm, C_norm, u_opt, elapsed, alpha, beta, gamma, eco_effect_percent)

        # ----- SAVE HISTORY -----
        A_history.append(A.copy())
        B_history.append(B_norm.copy())
        C_history.append(C_norm.copy())

        # ----- PLOTS -----
        update_buplot(ax_bu, utility_history, non_adapted_utility_history, t)
        canvas_bu.draw()

        update_fplot(ax_f, fitness_history, t, best_fitness)
        canvas_f.draw()

        update_matrix_plot(ax_A, A_history, t)
        canvas_A.draw()

        update_vector_plot(ax_B, B_history, t, "b")
        canvas_B.draw()

        update_vector_plot(ax_C, C_history, t, "c")
        canvas_C.draw()
        
        root.update()
        time.sleep(0.2)

# ================== ROOT ==================
root = tk.Tk()
root.title("Еволюційне моделювання слабконелінійних систем")
root.configure(bg=BG)
root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda e: root.destroy())

# ================== APP BAR ==================
appbar = tk.Frame(root, bg=CARD, height=56)
appbar.pack(fill=tk.X)

tk.Label(
    appbar,
    text="Еволюційне моделювання слабконелінійних систем",
    bg=CARD,
    fg=TEXT,
    font=FONT_TITLE
).pack(side=tk.LEFT, padx=24)

tk.Button(
    appbar,
    text="Запустити",
    bg=ACCENT,
    fg="black",
    font=FONT_UI,
    relief="flat",
    padx=18,
    pady=6,
    command=start
).pack(side=tk.RIGHT, padx=24)

tk.Button(
    appbar,
    text="Історія запусків",
    bg=BG,
    fg=TEXT,
    font=FONT_UI,
    relief="flat",
    padx=16,
    pady=6,
    command=lambda: open_history_window(root, load_run_config)
).pack(side=tk.RIGHT, padx=10)


# ================== CONTENT ==================
content = tk.Frame(root, bg=BG)
content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
content.grid_columnconfigure(0, weight=1)
content.grid_columnconfigure(1, weight=3)

# ================== LEFT ==================
left = tk.Frame(content, bg=BG)
left.grid(row=0, column=0, sticky="nsew", padx=(0, 20))

settings = card(left, "Параметри алгоритму")
settings.pack(fill=tk.X)

dimension_var = tk.IntVar(value=2)
population_var = tk.IntVar(value=50)
generations_var = tk.IntVar(value=50)
mutation_var = tk.DoubleVar(value=0.05)
disturbances_var = tk.IntVar(value=10)
disturbance_k_var = tk.DoubleVar(value=0.1)

dimension_var.trace_add("write", rebuild_matrices)

def spin(parent, label, var, frm, to):
    row = tk.Frame(parent, bg=CARD)
    row.pack(fill=tk.X, pady=6)
    tk.Label(row, text=label, bg=CARD, fg=MUTED).pack(side=tk.LEFT)
    tk.Spinbox(row, from_=frm, to=to, textvariable=var, width=7,
               bg=BG, fg=TEXT, insertbackground=TEXT,
               relief="flat", highlightthickness=1,
               highlightbackground=BORDER).pack(side=tk.RIGHT)

spin(settings, "Розмірність", dimension_var, 2, 6)
spin(settings, "Популяція", population_var, 10, 500)
spin(settings, "Покоління", generations_var, 10, 500)
spin(settings, "Кількість збурень T", disturbances_var, 1, 100)

tk.Label(settings, text="Ймовірність мутації", bg=CARD, fg=MUTED).pack(anchor="w")
tk.Scale(settings, from_=0.001, to=0.2, resolution=0.001,
         orient=tk.HORIZONTAL, variable=mutation_var,
         bg=CARD, troughcolor=BORDER,
         highlightthickness=0, fg=TEXT).pack(fill=tk.X)

tk.Label(settings, text="Параметр збурення k", bg=CARD, fg=MUTED).pack(anchor="w")
tk.Scale(settings, from_=0.05, to=0.5, resolution=0.01,
         orient=tk.HORIZONTAL, variable=disturbance_k_var,
         bg=CARD, troughcolor=BORDER,
         highlightthickness=0, fg=TEXT).pack(fill=tk.X)

# ===== SCENARIO =====
def set_scenario():
    global manual_scenario_data
    m = dimension_var.get()

    res = open_scenario_selection(root, m)
    if res == "UNCHANGED":
        # Користувач просто закрив вікно, нічого не змінюємо
        return 

    if res is None:
        # Користувач натиснув "Скинути"
        manual_scenario_data = None
        log_info("🎲 Режим сценарію вимкнено. Використовуються випадкові збурення.")
    else:
        manual_scenario_data = res
        log_info(f"✅ Активовано сценарій для m={m}")
        log_info(f"Alpha (C): {res['alpha']}")
        log_info(f"Beta (B): {res['beta']}")
        log_info(f"Gamma (A): {res['gamma']}")
    

tk.Button(settings, text="Налаштувати сценарій", bg=BR, fg=TEXT, 
          command=set_scenario).pack(fill=tk.X, pady=10)

log_card = card(left, "Журнал виконання")
log_card.pack(fill=tk.BOTH, expand=True, pady=(16, 0))

log_panel = tk.Text(
    log_card, bg=BG, fg=TEXT,
    font=FONT_MONO, relief="flat",
    insertbackground=TEXT,
    height=20
)
log_panel.pack(fill=tk.BOTH, expand=True)

# ===== RIGHT =====
def autofill_ABC():
    m = dimension_var.get()
    u0 = np.ones(m)
    A = np.random.uniform(0, 1, (m, m))
    q = np.random.dirichlet(np.ones(m))
    C = A @ u0
    B = q @ A

    for i in range(m):
        for j in range(m):
            A_entries[i][j].delete(0, tk.END)
            A_entries[i][j].insert(0, f"{A[i, j]:.4f}")

        B_entries[i].delete(0, tk.END)
        B_entries[i].insert(0, f"{B[i]:.4f}")

        C_entries[i].delete(0, tk.END)
        C_entries[i].insert(0, f"{C[i]:.4f}")

right = tk.Frame(content, bg=BG)
right.grid(row=0, column=1, sticky="nsew")

plot_card = card(right, "Візуалізація результатів")
plot_card.pack(fill=tk.BOTH, expand=True)

notebook = ttk.Notebook(plot_card)
notebook.pack(fill=tk.BOTH, expand=True)

# ===== HISTORY =====
def load_run_config(cfg):
    global current_loaded_cfg, replay_button

    """
    cfg = {
        "dimension": int,
        "A": np.array,
        "B": np.array,
        "C": np.array,
        "population": int,
        "generations": int,
        "mutation": float
    }
    """

    dimension_var.set(cfg["dimension"])
    population_var.set(cfg["population"])
    generations_var.set(cfg["generations"])
    mutation_var.set(cfg["mutation"])
    disturbances_var.set(cfg["disturbances"])
    disturbance_k_var.set(cfg["k"])

    rebuild_matrices()

    for i in range(cfg["dimension"]):
        for j in range(cfg["dimension"]):
            A_entries[i][j].delete(0, tk.END)
            A_entries[i][j].insert(0, f"{cfg['A'][i][j]:.4f}")

    for i in range(cfg["dimension"]):
        B_entries[i].delete(0, tk.END)
        B_entries[i].insert(0, f"{cfg['B'][i]:.4f}")

        C_entries[i].delete(0, tk.END)
        C_entries[i].insert(0, f"{cfg['C'][i]:.4f}")
    
    current_loaded_cfg = cfg
    if replay_button is None:
        replay_button = tk.Button(
            appbar,
            text="Відтворити експеримент",
            bg=BR,
            fg=TEXT,
            font=FONT_UI,
            relief="flat",
            padx=16,
            pady=6,
            command=lambda: replay_run(current_loaded_cfg)
        ).pack(side=tk.RIGHT, padx=10)


def replay_run(cfg):
    global replay_mode, replay_data

    replay_mode = True
    log_panel.delete("1.0", tk.END)

    log_info("▶ Відтворення експерименту")
    log_info(f"m = {cfg['dimension']}, T = {cfg['disturbances']}")

    fitness_history.clear()
    utility_history.clear()
    non_adapted_utility_history.clear()

    replay_data = get_evolution(cfg["run_id"])

    A_history, B_history, C_history = [], [], []

    for step in replay_data:
        t = step["t"]
        A = step["A"]
        B = step["B"]
        C = step["C"]
        u = step["u"]
        fitness = step["fitness"]
        utility = step["utility"]

        fitness_history.append(fitness)
        utility_history.append(utility)
        non_adapted_utility_history.append(non_adapted_utility)

        A_history.append(A)
        B_history.append(B)
        C_history.append(C)

        # ----- PLOTS -----
        update_buplot(ax_bu, utility_history, non_adapted_utility_history, t)
        update_fplot(ax_f, fitness_history, t, fitness)
        update_matrix_plot(ax_A, A_history, t)
        update_vector_plot(ax_B, B_history, t, "b")
        update_vector_plot(ax_C, C_history, t, "c")

        canvas_bu.draw()
        canvas_f.draw()
        canvas_A.draw()
        canvas_B.draw()
        canvas_C.draw()

        # ----- LOGS -----
        log_info(f"————————————— t = {t + 1} —————————————————")
        log_info(f"Найкраще значення цільової функції: {fitness_history[-1]:.4f}")
        log_info(f"Оптимальне u*: {u}")
        log_info(f"Оптимальна структура A*:\n{A}")

        root.update()
        time.sleep(0.3)

    log_info("■ Відтворення завершено")
    replay_mode = False

# ===== BU =====
bu_tab = tk.Frame(notebook, bg=CARD)
notebook.add(bu_tab, text="Пристосованість")

fig_bu, ax_bu = create_plot()
canvas_bu = FigureCanvasTkAgg(fig_bu, bu_tab)
canvas_bu.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ===== FITNESS =====
fitness_tab = tk.Frame(notebook, bg=CARD)
notebook.add(fitness_tab, text="Цільова функція")

fig_f, ax_f = create_plot()
canvas_f = FigureCanvasTkAgg(fig_f, fitness_tab)
canvas_f.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ===== A =====
A_tab = tk.Frame(notebook, bg=CARD)
notebook.add(A_tab, text="Матриця A")

fig_A, ax_A = create_plot()
canvas_A = FigureCanvasTkAgg(fig_A, A_tab)
canvas_A.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ===== B =====
B_tab = tk.Frame(notebook, bg=CARD)
notebook.add(B_tab, text="Вектор B")

fig_B, ax_B = create_plot()
canvas_B = FigureCanvasTkAgg(fig_B, B_tab)
canvas_B.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ===== C =====
C_tab = tk.Frame(notebook, bg=CARD)
notebook.add(C_tab, text="Вектор C")

fig_C, ax_C = create_plot()
canvas_C = FigureCanvasTkAgg(fig_C, C_tab)
canvas_C.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ===== VECTORS ======
matrices = card(right, "Вхідні дані (початкові)")
matrices.pack(fill=tk.X, pady=(16, 0))

A_frame = tk.Frame(matrices, bg=CARD)
B_frame = tk.Frame(matrices, bg=CARD)
C_frame = tk.Frame(matrices, bg=CARD)

A_frame.pack(side=tk.LEFT, padx=12)
B_frame.pack(side=tk.LEFT, padx=12)
C_frame.pack(side=tk.LEFT, padx=12)

A_entries = create_matrix(A_frame, 2, "A")
B_entries = create_vector(B_frame, 2, "B")
C_entries = create_vector(C_frame, 2, "C")

ui_ready = True

controls_frame = tk.Frame(matrices, bg=CARD)
controls_frame.pack(side=tk.RIGHT, padx=12, fill=tk.Y)

tk.Button(
    controls_frame,
    text="🎲 Автозаповнення",
    bg=BG, fg=TEXT, relief="flat",
    highlightthickness=1, highlightbackground=BORDER,
    padx=14, pady=6,
    command=autofill_ABC
).pack(fill=tk.X, pady=2)

tk.Button(
    controls_frame,
    text="📁 Завантажити з CSV",
    bg=BG, fg=ACCENT, relief="flat",
    highlightthickness=1, highlightbackground=BORDER,
    padx=14, pady=6,
    command=lambda: load_abc_from_csv(dimension_var.get(), A_entries, B_entries, C_entries)
).pack(fill=tk.X, pady=2)

root.mainloop()