"""
ACO_tsp.py — Ant Colony Optimization (ACO) para TSP.
Presupuesto: tiempo fijo (Opción A).
"""

import sys
import random
import time
from copy import deepcopy
from typing import List, Optional, Tuple

from utils import tour_length_idx


class AntColony:
    """
    ACO para TSP.

    Parámetros
    ----------
    dist       : matriz de distancias
    num_ants   : número de hormigas por iteración
    time_limit : segundos máximos de ejecución
    alpha      : peso de la feromona (τ)
    beta       : peso de la heurística (η = 1/d)
    rho        : tasa de evaporación
    Q          : constante de depósito
    tau0       : feromona inicial en todas las aristas
    elitist    : True → deposita solo la mejor global
    local_2opt : 2-opt a la mejor hormiga por iteración
    seed       : semilla para reproducibilidad
    """

    def __init__(
        self,
        dist: List[List[float]],
        num_ants: int       = 50,
        time_limit: float   = 10.0,
        alpha: float        = 1.0,
        beta: float         = 5.0,
        rho: float          = 0.1,
        Q: float            = 1.0,
        tau0: float         = 0.01,
        elitist: bool       = True,
        local_2opt: bool    = True,
        seed: Optional[int] = None,
    ):
        self.dist       = dist
        self.n          = len(dist)
        self.num_ants   = num_ants
        self.time_limit = time_limit
        self.alpha      = alpha
        self.beta       = beta
        self.rho        = rho
        self.Q          = Q
        self.tau0       = tau0
        self.elitist    = elitist
        self.local_2opt = local_2opt
        self.seed       = seed

        # Estado interno (disponible tras llamar a run())
        self.best_trail: Optional[List[int]]        = None
        self.best_len:   float                       = float("inf")
        self.history:    List[Tuple[float, float]]   = []
        self._pheromones: Optional[List[List[float]]] = None
        self._rng = random.Random(seed)

    # ── Feromonas ─────────────────────────────────────────────────────────

    def _init_pheromones(self) -> List[List[float]]:
        return [[self.tau0] * self.n for _ in range(self.n)]

    def _update_pheromones(self, ants: List[List[int]]) -> None:
        """Evaporación global + depósito elitista o por todas las hormigas."""
        for i in range(self.n):
            for j in range(self.n):
                self._pheromones[i][j] = max(
                    1e-10, self._pheromones[i][j] * (1.0 - self.rho)
                )
        if self.elitist:
            delta = self.Q / self.best_len
            for k in range(self.n):
                i = self.best_trail[k]
                j = self.best_trail[(k + 1) % self.n]
                self._pheromones[i][j] = min(self._pheromones[i][j] + delta, 1e5)
                self._pheromones[j][i] = self._pheromones[i][j]
        else:
            for trail in ants:
                cl    = tour_length_idx(trail, self.dist)
                delta = self.Q / cl
                for k in range(self.n):
                    i = trail[k]
                    j = trail[(k + 1) % self.n]
                    self._pheromones[i][j] = min(self._pheromones[i][j] + delta, 1e5)
                    self._pheromones[j][i] = self._pheromones[i][j]

    # ── Construcción de tour ──────────────────────────────────────────────

    def _move_probs(self, cityX: int, visited: List[bool]) -> List[float]:
        taueta = [0.0] * self.n
        total  = 0.0
        for i in range(self.n):
            if i == cityX or visited[i]:
                taueta[i] = 0.0
            else:
                eta = 1.0 / (self.dist[cityX][i] + 1e-12)
                val = (self._pheromones[cityX][i] ** self.alpha) * (eta ** self.beta)
                taueta[i] = max(1e-10, min(val, sys.float_info.max / (self.n * 100)))
            total += taueta[i]

        if total <= 0.0:
            not_vis = [i for i in range(self.n) if not visited[i] and i != cityX]
            probs   = [0.0] * self.n
            for i in not_vis:
                probs[i] = 1.0 / len(not_vis)
            return probs
        return [t / total for t in taueta]

    def _next_city(self, cityX: int, visited: List[bool]) -> int:
        probs = self._move_probs(cityX, visited)
        r, acc = self._rng.random(), 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        candidates = [i for i in range(self.n) if not visited[i] and i != cityX]
        return max(candidates, key=lambda i: probs[i])

    def _build_trail(self, start: int) -> List[int]:
        trail   = [0] * self.n
        visited = [False] * self.n
        trail[0]      = start
        visited[start] = True
        for i in range(self.n - 1):
            nxt          = self._next_city(trail[i], visited)
            trail[i + 1] = nxt
            visited[nxt] = True
        return trail

    def _two_opt(self, trail: List[int]) -> List[int]:
        best = trail[:]
        n    = len(best)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    d_before = self.dist[best[i-1]][best[i]] + self.dist[best[j]][best[(j+1) % n]]
                    d_after  = self.dist[best[i-1]][best[j]] + self.dist[best[i]][best[(j+1) % n]]
                    if d_after + 1e-10 < d_before:
                        best[i:j+1] = list(reversed(best[i:j+1]))
                        improved = True
        return best

    # ── Ciclo principal ───────────────────────────────────────────────────

    def run(self) -> Tuple[List[int], List[Tuple[float, float]]]:
        """
        Ejecuta el ACO durante time_limit segundos.

        Retorna
        -------
        (mejor_trail, historial)
        historial : lista de (tiempo_seg, mejor_longitud)
        """
        self._rng        = random.Random(self.seed)
        self._pheromones = self._init_pheromones()
        t_start          = time.perf_counter()
        deadline         = t_start + self.time_limit

        ants = [
            self._build_trail(self._rng.randint(0, self.n - 1))
            for _ in range(self.num_ants)
        ]
        self.best_trail = deepcopy(min(ants, key=lambda t: tour_length_idx(t, self.dist)))
        self.best_len   = tour_length_idx(self.best_trail, self.dist)
        self.history    = [(0.0, self.best_len)]

        while time.perf_counter() < deadline:
            ants = [
                self._build_trail(self._rng.randint(0, self.n - 1))
                for _ in range(self.num_ants)
            ]
            iter_best = min(ants, key=lambda t: tour_length_idx(t, self.dist))
            if self.local_2opt:
                iter_best = self._two_opt(iter_best)

            iter_len = tour_length_idx(iter_best, self.dist)
            if iter_len < self.best_len:
                self.best_len   = iter_len
                self.best_trail = deepcopy(iter_best)

            self._update_pheromones(ants)

            t_elapsed = time.perf_counter() - t_start
            self.history.append((round(t_elapsed, 4), self.best_len))

        return self.best_trail, self.history
