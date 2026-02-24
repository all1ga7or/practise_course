import numpy as np

class PenalizedModel:
    def __init__(self, A, B, C, u_min, u_max, k, penalty=100.0):
        self.A = A
        self.B = B
        self.C = C
        self.k = k
        self.M0 = penalty
        self.m = len(B)
        self.penalty_growth = 0.1

        self.A_min = (1 - k) * A
        self.A_max = (1 + k) * A
        self.u_min = u_min
        self.u_max = u_max

        self.generation = 0 

    # ---------- ОНОВЛЕННЯ ПОКОЛІННЯ ----------
    def set_generation(self, generation):
        self.generation = generation

    # ---------- ЦІЛЬОВА ФУНКЦІЯ ----------
    def objective(self, A, u):
        # Корисність
        utility = np.dot(self.B, u)

        # Обмеження Au = C
        residual = A @ u - self.C

        # Адаптивний штраф
        M = self.M0 * (1 + self.penalty_growth * self.generation)

        penalty = M * np.sum(residual ** 2)

        return utility - penalty