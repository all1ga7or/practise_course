import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from database import get_run_config, get_evolution


# =========================================================
# 3D Візуалізація простору рішень (m = 3)
# =========================================================
def build_solution_surface(run_id, sample_points=2000):

    cfg = get_run_config(run_id)
    evolution = get_evolution(run_id)

    if cfg is None:
        print("Run not found")
        return

    if not evolution:
        print("No evolution data found")
        return

    # ---- беремо останній стан ----
    last_state = evolution[-1]

    A = np.array(last_state["A"])
    B = np.array(last_state["B"])
    C = np.array(last_state["C"])

    u_min = np.array(cfg["u_min"])
    u_max = np.array(cfg["u_max"])

    if len(u_min) != 3:
        print("3D visualization available only for m = 3")
        return

    # ---- створюємо папку ----
    save_dir = "results/evolutions"
    os.makedirs(save_dir, exist_ok=True)

    # =====================================================
    # 1️⃣ Генерація області пошуку
    # =====================================================
    U = np.random.uniform(u_min, u_max, (sample_points, 3))

    M = 1000
    F_vals = []

    for u in U:
        penalty = np.sum(np.abs(A @ u - C))
        F = -np.dot(B, u) + M * penalty
        F_vals.append(F)

    F_vals = np.array(F_vals)

    # =====================================================
    # 2️⃣ Побудова 3D графіка
    # =====================================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        U[:, 0],
        U[:, 1],
        U[:, 2],
        c=F_vals,
        cmap="viridis",
        alpha=0.4
    )

    plt.colorbar(scatter, ax=ax, label="F(u)")

    # =====================================================
    # 3️⃣ Траєкторія еволюції
    # =====================================================
    trajectory = np.array([step["u"] for step in evolution])

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        color="black",
        linewidth=2,
        label="Траєкторія еволюції"
    )

    # =====================================================
    # 4️⃣ Оптимальне рішення (останній крок)
    # =====================================================
    best_u = trajectory[-1]

    ax.scatter(
        best_u[0],
        best_u[1],
        best_u[2],
        color="red",
        s=120,
        label="Оптимум u*"
    )

    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_zlabel("u3")
    ax.set_title("3D Простір рішень та еволюція оптимізації")

    ax.legend()

    filename = os.path.join(save_dir, f"surface3D_run_{run_id}.png")
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"3D visualization saved to {filename}")

if __name__ == "__main__":
    build_solution_surface(run_id=73)

