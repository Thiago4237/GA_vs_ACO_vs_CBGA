"""
utils.py — Funciones compartidas para todos los algoritmos del taller.

Responsabilidades:
    - Leer instancias TSPLIB (.tsp, EUC_2D)
    - Calcular distancias y evaluar tours
"""

import math
from typing import List

City = complex  # ciudad representada como número complejo (x + yj)
Tour = List[City]


def read_tsp(filepath: str) -> List[City]:
    """
    Lee un archivo .tsp en formato TSPLIB (NODE_COORD_SECTION EUC_2D).
    Devuelve lista de City (complex).
    """
    cities = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    reading = False
    for line in lines:
        line = line.strip()
        if line == "NODE_COORD_SECTION":
            reading = True
            continue
        if line == "EOF":
            break
        if reading:
            parts = line.split()
            if len(parts) >= 3:
                _, x, y = parts[0], parts[1], parts[2]
                cities.append(City(float(x), float(y)))

    return cities


def make_dist_matrix(cities: List[City]) -> List[List[float]]:
    """
    Construye la matriz de distancias euclídeas entre ciudades.
    Usa redondeo NINT como especifica TSPLIB para EUC_2D.
    """
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = round(abs(cities[i] - cities[j]))
    return dist


def tour_length(tour: Tour) -> float:
    """
    Longitud de un tour dado como lista de City (ciclo cerrado).
    Compatible con GA que trabaja con listas de City.
    """
    return sum(abs(tour[i] - tour[i - 1]) for i in range(len(tour)))


def tour_length_idx(trail: List[int], dist: List[List[float]]) -> float:
    """
    Longitud de un tour dado como lista de índices (ciclo cerrado).
    Compatible con ACO que trabaja con índices enteros.
    """
    n = len(trail)
    return sum(dist[trail[i]][trail[(i + 1) % n]] for i in range(n))