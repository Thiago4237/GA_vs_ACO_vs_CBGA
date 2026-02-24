"""
ga_tsp.py — Algoritmo Genético (GA) para TSP.
Presupuesto: por tiempo (Opción A).
Retorna: (mejor_tour, historial_convergencia)
"""

import random
import time
from typing import List, Optional, Tuple

from utils import City, Tour, tour_length


def _random_individual(city_list, rng):
    ind = city_list[:]
    rng.shuffle(ind)
    return ind


def _order_crossover(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[a:b] = p1[a:b]
    fill = [c for c in p2 if c not in child]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child


def _swap_mutation(tour, rng):
    mutated = tour[:]
    i, j = rng.sample(range(len(mutated)), 2)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def _tournament_selection(population, k, rng):
    candidates = rng.sample(population, k)
    return min(candidates, key=tour_length)


def _two_opt(tour):
    best = tour[:]
    n = len(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                d_before = abs(best[i-1] - best[i])   + abs(best[j] - best[(j+1) % n])
                d_after  = abs(best[i-1] - best[j])   + abs(best[i] - best[(j+1) % n])
                if d_after + 1e-10 < d_before:
                    best[i:j+1] = list(reversed(best[i:j+1]))
                    improved = True
    return best


def genetic_tsp(
    cities: List[City],
    pop_size: int,
    time_limit: float,
    pc: float,
    pm: float,
    elitism_k: int,
    tournament_k: int,
    local_2opt: bool = True,
    seed: Optional[int] = None,
) -> Tuple[Tour, List[Tuple[float, float]]]:
    """
    Retorna
    -------
    (mejor_tour, historial)
    historial: lista de (tiempo_seg, mejor_longitud) — best-so-far en el tiempo
    """
    rng      = random.Random(seed)
    t_start  = time.perf_counter()
    deadline = t_start + time_limit
    city_list = list(cities)

    population = [_random_individual(city_list, rng) for _ in range(pop_size)]
    best_tour  = min(population, key=tour_length)
    best_len   = tour_length(best_tour)

    historial: List[Tuple[float, float]] = [(0.0, best_len)]

    while time.perf_counter() < deadline:
        population.sort(key=tour_length)
        next_gen = population[:elitism_k]

        while len(next_gen) < pop_size:
            p1 = _tournament_selection(population, tournament_k, rng)
            p2 = _tournament_selection(population, tournament_k, rng)
            child = _order_crossover(p1, p2, rng) if rng.random() < pc else p1[:]
            if rng.random() < pm:
                child = _swap_mutation(child, rng)
            next_gen.append(child)

        population = next_gen
        gen_best = min(population, key=tour_length)
        if local_2opt:
            gen_best = _two_opt(gen_best)

        gen_len = tour_length(gen_best)
        if gen_len < best_len:
            best_len  = gen_len
            best_tour = gen_best[:]

        # Registrar punto de convergencia
        t_elapsed = time.perf_counter() - t_start
        historial.append((round(t_elapsed, 4), best_len))

    return best_tour, historial