from matplotlib.figure import Figure
import numpy as np

def create_plot():
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Час")
    ax.set_ylabel("Значення")
    ax.grid(True)
    return fig, ax

def update_fplot(ax, data, gen, fitness):
    ax.clear()
    ax.plot(data, linestyle="-", color="blue", label="f")
    ax.set_title(f"Еволюція цільової функції")
    ax.set_xlabel("Час")
    ax.set_ylabel("Значення цільової функції f")
    ax.legend()
    ax.grid(True)

    ax.text(
        0.98, 0.02,
        f"Поточне значення: {fitness:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom"
    )

def update_buplot(ax, adapted_hist, base_hist, t):
    ax.clear()
    steps = range(1, len(adapted_hist) + 1)
    
    # Лінія адаптованого прибутку
    ax.plot(steps, adapted_hist, color='#38bdf8', linewidth=2, label='Bu*')
    
    # Лінія базового прибутку
    ax.plot(steps, base_hist, color='#f87171', linestyle='--', linewidth=1.5, label='Bu0')
    
    ax.set_title("Еволюція прибутку")
    ax.set_xlabel("Крок збурення (t)")
    ax.set_ylabel("млрд грн")
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.2)

def update_vector_plot(ax, history, gen, label):
    ax.clear()
    history = np.array(history)

    for i in range(history.shape[1]):
        ax.plot(history[:, i], label=f"{label}{i+1}")

    ax.set_xlabel("Час")
    ax.set_ylabel("Значення")
    ax.legend()
    ax.grid(True)


def update_matrix_plot(ax, history, gen):
    ax.clear()
    history = np.array(history)

    m = history.shape[1]
    for i in range(m):
        for j in range(m):
            ax.plot(history[:, i, j], label=f"a{i+1}{j+1}")

    ax.set_xlabel("Час")
    ax.set_ylabel("Значення")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)


