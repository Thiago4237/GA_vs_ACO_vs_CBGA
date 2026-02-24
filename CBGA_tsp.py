"""
cbga_tsp.py — Algoritmo Genético de Chu-Beasley (CBGA) para TSP.
Presupuesto: por tiempo (Opción A).
Retorna: (mejor_tour, historial_convergencia)
"""

import random
import time
from typing import List, Optional, Tuple

from utils import City, Tour, tour_length


def _edge_set(tour):
    n = len(tour)
    edges = set()
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        if (a.real, a.imag) < (b.real, b.imag):
            edges.add((a, b))
        else:
            edges.add((b, a))
    return frozenset(edges)


def _diversity(tour_a, tour_b):
    ea = _edge_set(tour_a)
    eb = _edge_set(tour_b)
    return 1.0 - len(ea & eb) / len(ea)


def _is_duplicate(tour, population):
    ea = _edge_set(tour)
    return any(_edge_set(ind) == ea for ind in population)


def _min_diversity_to_pop(tour, population):
    return min(_diversity(tour, ind) for ind in population)


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


def _select_replacement(child, population, min_diversity):
    similar = [i for i, ind in enumerate(population)
               if _diversity(child, ind) < min_diversity]
    if similar:
        worst_idx = max(similar, key=lambda i: tour_length(population[i]))
    else:
        worst_idx = max(range(len(population)), key=lambda i: tour_length(population[i]))
    if tour_length(child) < tour_length(population[worst_idx]):
        return worst_idx
    return None


def cbga_tsp(
    cities: List[City],
    pop_size: int,
    time_limit: float,
    pc: float,
    pm: float,
    tournament_k: int,
    min_diversity: float = 0.15,
    attempts_per_gen: int = 5,
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

    population: List[Tour] = []
    attempts = 0
    while len(population) < pop_size and attempts < pop_size * 20:
        ind = _random_individual(city_list, rng)
        if not _is_duplicate(ind, population):
            population.append(ind)
        attempts += 1

    best_tour = min(population, key=tour_length)
    best_len  = tour_length(best_tour)

    historial: List[Tuple[float, float]] = [(0.0, best_len)]

    while time.perf_counter() < deadline:
        for _ in range(attempts_per_gen):
            p1 = _tournament_selection(population, tournament_k, rng)
            p2 = _tournament_selection(population, tournament_k, rng)
            child = _order_crossover(p1, p2, rng) if rng.random() < pc else p1[:]
            if rng.random() < pm:
                child = _swap_mutation(child, rng)
            if _is_duplicate(child, population):
                continue
            if _min_diversity_to_pop(child, population) < min_diversity:
                continue
            idx = _select_replacement(child, population, min_diversity)
            if idx is not None:
                population[idx] = child

        gen_best = min(population, key=tour_length)
        if local_2opt:
            gen_best = _two_opt(gen_best)
        gen_len = tour_length(gen_best)
        if gen_len < best_len:
            best_len  = gen_len
            best_tour = gen_best[:]

        t_elapsed = time.perf_counter() - t_start
        historial.append((round(t_elapsed, 4), best_len))

    return best_tour, historial