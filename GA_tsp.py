"""
GA_tsp.py — Algoritmo Genético (GA) para TSP.
Presupuesto: tiempo fijo (Opción A).
"""

import random
import time
from typing import List, Optional, Tuple

from utils import City, Tour, tour_length


class GeneticAlgorithm:
    """
    Algoritmo Genético clásico para TSP.

    Parámetros
    ----------
    cities       : lista de ciudades (City = complex)
    pop_size     : tamaño de la población
    time_limit   : segundos máximos de ejecución
    pc           : probabilidad de cruce
    pm           : probabilidad de mutación
    elitism_k    : individuos élite por generación
    tournament_k : tamaño del torneo de selección
    local_2opt   : aplica 2-opt al mejor individuo por generación
    seed         : semilla para reproducibilidad
    """

    def __init__(
        self,
        cities: List[City],
        pop_size: int       = 50,
        time_limit: float   = 10.0,
        pc: float           = 0.9,
        pm: float           = 0.1,
        elitism_k: int      = 5,
        tournament_k: int   = 3,
        local_2opt: bool    = True,
        seed: Optional[int] = None,
    ):
        self.cities       = list(cities)
        self.pop_size     = pop_size
        self.time_limit   = time_limit
        self.pc           = pc
        self.pm           = pm
        self.elitism_k    = elitism_k
        self.tournament_k = tournament_k
        self.local_2opt   = local_2opt
        self.seed         = seed

        # Estado interno (disponible tras llamar a run())
        self.best_tour: Optional[Tour]          = None
        self.best_len:  float                   = float("inf")
        self.history:   List[Tuple[float,float]] = []
        self._rng = random.Random(seed)

    # ── Operadores genéticos ──────────────────────────────────────────────

    def _random_individual(self) -> Tour:
        ind = self.cities[:]
        self._rng.shuffle(ind)
        return ind

    def _order_crossover(self, p1: Tour, p2: Tour) -> Tour:
        """OX — garantiza permutación válida."""
        n = len(p1)
        a, b = sorted(self._rng.sample(range(n), 2))
        child = [None] * n
        child[a:b] = p1[a:b]
        fill = [c for c in p2 if c not in child]
        idx = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        return child

    def _swap_mutation(self, tour: Tour) -> Tour:
        mutated = tour[:]
        i, j = self._rng.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def _tournament_selection(self, population: List[Tour]) -> Tour:
        candidates = self._rng.sample(population, self.tournament_k)
        return min(candidates, key=tour_length)

    def _two_opt(self, tour: Tour) -> Tour:
        best = tour[:]
        n = len(best)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    d_before = abs(best[i-1] - best[i])  + abs(best[j] - best[(j+1) % n])
                    d_after  = abs(best[i-1] - best[j])  + abs(best[i] - best[(j+1) % n])
                    if d_after + 1e-10 < d_before:
                        best[i:j+1] = list(reversed(best[i:j+1]))
                        improved = True
        return best

    # ── Ciclo principal ───────────────────────────────────────────────────

    def run(self) -> Tuple[Tour, List[Tuple[float, float]]]:
        """
        Ejecuta el GA durante time_limit segundos.

        Retorna
        -------
        (mejor_tour, historial)
        historial : lista de (tiempo_seg, mejor_longitud)
        """
        self._rng    = random.Random(self.seed)
        t_start      = time.perf_counter()
        deadline     = t_start + self.time_limit

        population   = [self._random_individual() for _ in range(self.pop_size)]
        self.best_tour = min(population, key=tour_length)
        self.best_len  = tour_length(self.best_tour)
        self.history   = [(0.0, self.best_len)]

        while time.perf_counter() < deadline:
            population.sort(key=tour_length)
            next_gen = population[:self.elitism_k]

            while len(next_gen) < self.pop_size:
                p1 = self._tournament_selection(population)
                p2 = self._tournament_selection(population)
                child = self._order_crossover(p1, p2) if self._rng.random() < self.pc else p1[:]
                if self._rng.random() < self.pm:
                    child = self._swap_mutation(child)
                next_gen.append(child)

            population = next_gen
            gen_best   = min(population, key=tour_length)
            if self.local_2opt:
                gen_best = self._two_opt(gen_best)

            gen_len = tour_length(gen_best)
            if gen_len < self.best_len:
                self.best_len  = gen_len
                self.best_tour = gen_best[:]

            t_elapsed = time.perf_counter() - t_start
            self.history.append((round(t_elapsed, 4), self.best_len))

        return self.best_tour, self.history
