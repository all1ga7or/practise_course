import numpy as np
import pygad


class GeneticAlgorithm:
    def __init__(self, model, pop_size, generations, mutation_rate):
        self.model = model
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        self.m = model.A.shape[0]

        # Кількість генів: A (m*m) + u (m)
        self.num_genes = self.m * self.m + self.m

        self.gene_space = []

        # ---------- A_ij ----------
        for i in range(self.m):
            for j in range(self.m):
                self.gene_space.append({
                    "low": float(model.A_min[i, j]),
                    "high": float(model.A_max[i, j])
                })

        # ---------- u_j ----------
        for j in range(self.m):
            self.gene_space.append({
                "low": float(model.u_min[j]),
                "high": float(model.u_max[j])
            })

        self.best_solution = None
        self.best_fitness = None

    # ---------- FITNESS ----------
    def fitness_func(self, ga_instance, solution, solution_idx):
        A_flat = solution[: self.m * self.m]
        u = solution[self.m * self.m :]

        A = A_flat.reshape((self.m, self.m))

        return float(self.model.objective(A, u))  

    # ---------- CALLBACK ----------
    def on_generation(self, ga_instance):
        self.model.set_generation(ga_instance.generations_completed)
        solution, fitness, _ = ga_instance.best_solution()
        self.best_solution = solution
        self.best_fitness = fitness

    # ---------- RUN ----------
    def run(self, last_best_solution=None):
        initial_pop = None
        if last_best_solution is not None:
            initial_pop = np.tile(last_best_solution, (self.pop_size, 1))
            initial_pop += np.random.normal(0, 0.02, initial_pop.shape)

        ga = pygad.GA(
            num_generations=self.generations,
            sol_per_pop=self.pop_size,
            initial_population=initial_pop,
            num_parents_mating=max(2, self.pop_size // 4),
            num_genes=self.num_genes,
            gene_space=self.gene_space,
            fitness_func=self.fitness_func,
            mutation_percent_genes=[10, 2],
            parent_selection_type="rank",
            crossover_type="uniform",
            mutation_type="adaptive",
            keep_elitism=5,
            on_generation=self.on_generation,
            parallel_processing=["thread", 8],
            stop_criteria=["reach_1000", "saturate_15"]
        )

        ga.run()

        solution, fitness, _ = ga.best_solution()

        A_flat = solution[: self.m * self.m]
        u = solution[self.m * self.m :]

        A = A_flat.reshape((self.m, self.m))

        return fitness, (A, u)
