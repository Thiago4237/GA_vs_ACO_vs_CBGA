"""
aco_tsp.py — Ant Colony Optimization (ACO) para TSP.
Presupuesto: por tiempo (Opción A).
Retorna: (mejor_trail, historial_convergencia)
"""

import sys
import random
import time
from copy import deepcopy
from typing import List, Optional, Tuple

from utils import tour_length_idx


def _init_pheromones(n, tau0):
    return [[tau0] * n for _ in range(n)]


def _move_probs(cityX, visited, pheromones, dist, alpha, beta):
    n = len(pheromones)
    taueta = [0.0] * n
    total = 0.0
    for i in range(n):
        if i == cityX or visited[i]:
            taueta[i] = 0.0
        else:
            eta = 1.0 / (dist[cityX][i] + 1e-12)
            val = (pheromones[cityX][i] ** alpha) * (eta ** beta)
            taueta[i] = max(1e-10, min(val, sys.float_info.max / (n * 100)))
        total += taueta[i]
    if total <= 0.0:
        not_vis = [i for i in range(n) if not visited[i] and i != cityX]
        probs = [0.0] * n
        for i in not_vis:
            probs[i] = 1.0 / len(not_vis)
        return probs
    return [t / total for t in taueta]


def _next_city(cityX, visited, pheromones, dist, alpha, beta, rng):
    probs = _move_probs(cityX, visited, pheromones, dist, alpha, beta)
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    candidates = [i for i in range(len(probs)) if not visited[i] and i != cityX]
    return max(candidates, key=lambda i: probs[i])


def _two_opt(trail, dist):
    best = trail[:]
    n = len(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                d_before = dist[best[i-1]][best[i]] + dist[best[j]][best[(j+1) % n]]
                d_after  = dist[best[i-1]][best[j]] + dist[best[i]][best[(j+1) % n]]
                if d_after + 1e-10 < d_before:
                    best[i:j+1] = list(reversed(best[i:j+1]))
                    improved = True
    return best


def _build_trail(start, pheromones, dist, alpha, beta, rng):
    n = len(pheromones)
    trail = [0] * n
    visited = [False] * n
    trail[0] = start
    visited[start] = True
    for i in range(n - 1):
        nxt = _next_city(trail[i], visited, pheromones, dist, alpha, beta, rng)
        trail[i + 1] = nxt
        visited[nxt] = True
    return trail


def _update_pheromones(pheromones, ants, dist, best_trail, best_len, rho, Q, elitist):
    n = len(pheromones)
    for i in range(n):
        for j in range(n):
            pheromones[i][j] = max(1e-10, pheromones[i][j] * (1.0 - rho))
    if elitist:
        delta = Q / best_len
        m = len(best_trail)
        for k in range(m):
            i = best_trail[k]
            j = best_trail[(k + 1) % m]
            pheromones[i][j] = min(pheromones[i][j] + delta, 1e5)
            pheromones[j][i] = pheromones[i][j]
    else:
        for trail in ants:
            cl = tour_length_idx(trail, dist)
            delta = Q / cl
            m = len(trail)
            for k in range(m):
                i = trail[k]
                j = trail[(k + 1) % m]
                pheromones[i][j] = min(pheromones[i][j] + delta, 1e5)
                pheromones[j][i] = pheromones[i][j]


def ant_colony_tsp(
    dist: List[List[float]],
    num_ants: int,
    time_limit: float,
    alpha: float,
    beta: float,
    rho: float,
    Q: float,
    tau0: float,
    elitist: bool,
    local_2opt: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[int], List[Tuple[float, float]]]:
    """
    Retorna
    -------
    (mejor_trail, historial)
    historial: lista de (tiempo_seg, mejor_longitud) — best-so-far en el tiempo
    """
    rng      = random.Random(seed)
    t_start  = time.perf_counter()
    deadline = t_start + time_limit
    n        = len(dist)
    pheromones = _init_pheromones(n, tau0)

    ants = [
        _build_trail(rng.randint(0, n-1), pheromones, dist, alpha, beta, rng)
        for _ in range(num_ants)
    ]
    best_trail = deepcopy(min(ants, key=lambda t: tour_length_idx(t, dist)))
    best_len   = tour_length_idx(best_trail, dist)

    historial: List[Tuple[float, float]] = [(0.0, best_len)]

    while time.perf_counter() < deadline:
        ants = [
            _build_trail(rng.randint(0, n-1), pheromones, dist, alpha, beta, rng)
            for _ in range(num_ants)
        ]
        iter_best = min(ants, key=lambda t: tour_length_idx(t, dist))
        if local_2opt:
            iter_best = _two_opt(iter_best, dist)

        iter_best_len = tour_length_idx(iter_best, dist)
        if iter_best_len < best_len:
            best_len   = iter_best_len
            best_trail = deepcopy(iter_best)

        _update_pheromones(pheromones, ants, dist, best_trail, best_len,
                           rho, Q, elitist)

        t_elapsed = time.perf_counter() - t_start
        historial.append((round(t_elapsed, 4), best_len))

    return best_trail, historial