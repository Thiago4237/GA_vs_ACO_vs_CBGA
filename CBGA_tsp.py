"""
CBGA_tsp.py — Algoritmo Genético de Chu-Beasley (CBGA) para TSP.
Implementación orientada a objetos (OOP).
Presupuesto: tiempo fijo (Opción A).
"""

import random
import time
from typing import List, Optional, Tuple

from utils import City, Tour, tour_length


class ChuBeasleyGA:
    """
    Algoritmo Genético de Chu-Beasley para TSP.

    Diferencias clave frente al GA clásico:
      1. Sin duplicados en la población (hash por aristas)
      2. Control explícito de diversidad (fracción de aristas distintas)
      3. Reemplazo selectivo: el hijo reemplaza al peor individuo similar
      4. Múltiples intentos de reproducción por ciclo

    Parámetros
    ----------
    cities           : lista de ciudades (City = complex)
    pop_size         : tamaño de la población
    time_limit       : segundos máximos de ejecución
    pc               : probabilidad de cruce
    pm               : probabilidad de mutación
    tournament_k     : tamaño del torneo de selección
    min_diversity    : fracción mínima de aristas distintas (0.0–1.0)
    attempts_per_gen : intentos de reproducción por ciclo
    local_2opt       : aplica 2-opt al mejor individuo por ciclo
    seed             : semilla para reproducibilidad
    """

    def __init__(
        self,
        cities: List[City],
        pop_size: int         = 50,
        time_limit: float     = 10.0,
        pc: float             = 0.9,
        pm: float             = 0.1,
        tournament_k: int     = 3,
        min_diversity: float  = 0.15,
        attempts_per_gen: int = 5,
        local_2opt: bool      = True,
        seed: Optional[int]   = None,
    ):
        self.cities           = list(cities)
        self.pop_size         = pop_size
        self.time_limit       = time_limit
        self.pc               = pc
        self.pm               = pm
        self.tournament_k     = tournament_k
        self.min_diversity    = min_diversity
        self.attempts_per_gen = attempts_per_gen
        self.local_2opt       = local_2opt
        self.seed             = seed

        # Estado interno (disponible tras llamar a run())
        self.best_tour: Optional[Tour]           = None
        self.best_len:  float                    = float("inf")
        self.history:   List[Tuple[float, float]] = []
        self._rng = random.Random(seed)

    # ── Diversidad basada en aristas ──────────────────────────────────────

    def _edge_set(self, tour: Tour) -> frozenset:
        """Representa un tour como conjunto de aristas sin dirección ni origen."""
        n = len(tour)
        edges = set()
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]
            edges.add((a, b) if (a.real, a.imag) < (b.real, b.imag) else (b, a))
        return frozenset(edges)

    def _diversity(self, tour_a: Tour, tour_b: Tour) -> float:
        """Fracción de aristas NO compartidas entre dos tours (0=idénticos, 1=distintos)."""
        ea = self._edge_set(tour_a)
        eb = self._edge_set(tour_b)
        return 1.0 - len(ea & eb) / len(ea)

    def _is_duplicate(self, tour: Tour, population: List[Tour]) -> bool:
        ea = self._edge_set(tour)
        return any(self._edge_set(ind) == ea for ind in population)

    def _min_diversity_to_pop(self, tour: Tour, population: List[Tour]) -> float:
        return min(self._diversity(tour, ind) for ind in population)

    # ── Operadores genéticos ──────────────────────────────────────────────

    def _random_individual(self) -> Tour:
        ind = self.cities[:]
        self._rng.shuffle(ind)
        return ind

    def _order_crossover(self, p1: Tour, p2: Tour) -> Tour:
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

    # ── Reemplazo selectivo ───────────────────────────────────────────────

    def _select_replacement(self, child: Tour, population: List[Tour]) -> Optional[int]:
        """
        Política CBGA: el hijo compite contra el peor individuo similar.
        Si no hay similares, compite contra el peor global.
        """
        similar = [
            i for i, ind in enumerate(population)
            if self._diversity(child, ind) < self.min_diversity
        ]
        worst_idx = (
            max(similar, key=lambda i: tour_length(population[i]))
            if similar else
            max(range(len(population)), key=lambda i: tour_length(population[i]))
        )
        return worst_idx if tour_length(child) < tour_length(population[worst_idx]) else None

    # ── Ciclo principal ───────────────────────────────────────────────────

    def run(self) -> Tuple[Tour, List[Tuple[float, float]]]:
        """
        Ejecuta el CBGA durante time_limit segundos.

        Retorna
        -------
        (mejor_tour, historial)
        historial : lista de (tiempo_seg, mejor_longitud)
        """
        self._rng = random.Random(self.seed)
        t_start   = time.perf_counter()
        deadline  = t_start + self.time_limit

        # Inicialización sin duplicados
        population: List[Tour] = []
        attempts = 0
        while len(population) < self.pop_size and attempts < self.pop_size * 20:
            ind = self._random_individual()
            if not self._is_duplicate(ind, population):
                population.append(ind)
            attempts += 1

        self.best_tour = min(population, key=tour_length)
        self.best_len  = tour_length(self.best_tour)
        self.history   = [(0.0, self.best_len)]

        while time.perf_counter() < deadline:
            for _ in range(self.attempts_per_gen):
                p1 = self._tournament_selection(population)
                p2 = self._tournament_selection(population)
                child = self._order_crossover(p1, p2) if self._rng.random() < self.pc else p1[:]
                if self._rng.random() < self.pm:
                    child = self._swap_mutation(child)

                if self._is_duplicate(child, population):
                    continue
                if self._min_diversity_to_pop(child, population) < self.min_diversity:
                    continue

                idx = self._select_replacement(child, population)
                if idx is not None:
                    population[idx] = child

            gen_best = min(population, key=tour_length)
            if self.local_2opt:
                gen_best = self._two_opt(gen_best)

            gen_len = tour_length(gen_best)
            if gen_len < self.best_len:
                self.best_len  = gen_len
                self.best_tour = gen_best[:]

            t_elapsed = time.perf_counter() - t_start
            self.history.append((round(t_elapsed, 4), self.best_len))

        return self.best_tour, self.history


# ── Función wrapper — mantiene compatibilidad con main.py y Runner.py ─────────

def cbga_tsp(
    cities: List[City],
    pop_size: int,
    time_limit: float,
    pc: float,
    pm: float,
    tournament_k: int,
    min_diversity: float  = 0.15,
    attempts_per_gen: int = 5,
    local_2opt: bool      = True,
    seed: Optional[int]   = None,
) -> Tuple[Tour, List[Tuple[float, float]]]:
    """Wrapper funcional sobre ChuBeasleyGA para compatibilidad."""
    algo = ChuBeasleyGA(
        cities=cities, pop_size=pop_size, time_limit=time_limit,
        pc=pc, pm=pm, tournament_k=tournament_k,
        min_diversity=min_diversity, attempts_per_gen=attempts_per_gen,
        local_2opt=local_2opt, seed=seed,
    )
    return algo.run()