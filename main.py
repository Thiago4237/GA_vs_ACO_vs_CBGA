"""
main.py — Punto de entrada del taller.
Presupuesto: Opción A (tiempo fijo T segundos por algoritmo).

Uso:
    python main.py --instance data/berlin52.tsp
    python main.py --instance data/berlin52.tsp --time-limit 10
"""

import time
import argparse
import os

from utils    import read_tsp, make_dist_matrix, tour_length, tour_length_idx
from GA_tsp   import GeneticAlgorithm
from ACO_tsp  import AntColony
from CBGA_tsp import ChuBeasleyGA

OPTIMOS = {
    "berlin52.tsp": 7542,
    "eil51.tsp":    426,
    "att48.tsp":    10628,
    "st70.tsp":     675,
}


class Comparador:
    """
    Ejecuta y compara GA, CBGA y ACO sobre una instancia TSP.

    Parámetros
    ----------
    args    : namespace de argparse con todos los hiperparámetros
    cities  : lista de ciudades leída desde el .tsp
    dist    : matriz de distancias
    nombre  : nombre del archivo de instancia
    """

    def __init__(self, args, cities, dist, nombre):
        self.args    = args
        self.cities  = cities
        self.dist    = dist
        self.nombre  = nombre
        self.optimo  = OPTIMOS.get(nombre)
        self.use_2opt = not args.no_2opt
        self.results  = {}

    def _gap_str(self, length: float) -> str:
        if self.optimo:
            return f"  GAP={(length - self.optimo) / self.optimo * 100:.2f}%"
        return ""

    def _mostrar(self, algo: str, length: float, t: float) -> None:
        print(f"Resultado  : {length:.1f}  ({t:.2f}s){self._gap_str(length)}\n")

    def run_ga(self) -> None:
        args = self.args
        print("--- GA ---")
        print(f"pop={args.pop}  pc={args.pc}  pm={args.pm}  "
              f"élite={args.elitism}  torneo={args.tournament}  2opt={self.use_2opt}")

        algo = GeneticAlgorithm(
            cities=self.cities, pop_size=args.pop, time_limit=args.time_limit,
            pc=args.pc, pm=args.pm, elitism_k=args.elitism,
            tournament_k=args.tournament, local_2opt=self.use_2opt, seed=args.seed,
        )
        t0 = time.perf_counter()
        tour, _ = algo.run()
        elapsed = time.perf_counter() - t0
        length  = tour_length(tour)
        self.results["GA"] = (length, elapsed)
        self._mostrar("GA", length, elapsed)

    def run_cbga(self) -> None:
        args = self.args
        print("--- CBGA ---")
        print(f"pop={args.pop}  pc={args.pc}  pm={args.pm}  torneo={args.tournament}  "
              f"div={args.diversity}  intentos={args.attempts}  2opt={self.use_2opt}")

        algo = ChuBeasleyGA(
            cities=self.cities, pop_size=args.pop, time_limit=args.time_limit,
            pc=args.pc, pm=args.pm, tournament_k=args.tournament,
            min_diversity=args.diversity, attempts_per_gen=args.attempts,
            local_2opt=self.use_2opt, seed=args.seed,
        )
        t0 = time.perf_counter()
        tour, _ = algo.run()
        elapsed = time.perf_counter() - t0
        length  = tour_length(tour)
        self.results["CBGA"] = (length, elapsed)
        self._mostrar("CBGA", length, elapsed)

    def run_aco(self) -> None:
        args = self.args
        print("--- ACO ---")
        print(f"ants={args.ants}  α={args.alpha}  β={args.beta}  ρ={args.rho}  "
              f"Q={args.q}  elitist={not args.no_elitist}  2opt={self.use_2opt}")

        algo = AntColony(
            dist=self.dist, num_ants=args.ants, time_limit=args.time_limit,
            alpha=args.alpha, beta=args.beta, rho=args.rho, Q=args.q,
            tau0=args.tau0, elitist=not args.no_elitist,
            local_2opt=self.use_2opt, seed=args.seed,
        )
        t0 = time.perf_counter()
        trail, _ = algo.run()
        elapsed  = time.perf_counter() - t0
        length   = tour_length_idx(trail, self.dist)
        self.results["ACO"] = (length, elapsed)
        self._mostrar("ACO", length, elapsed)

    def print_resumen(self) -> None:
        linea = "=" * 56
        print(linea)
        header = f"{'Algoritmo':<10} {'Longitud':>12} {'Tiempo (s)':>12}"
        if self.optimo:
            header += f" {'GAP%':>8}"
        print(header)
        print("-" * 46)
        for algo, (length, t) in self.results.items():
            row = f"{algo:<10} {length:>12.1f} {t:>12.2f}"
            if self.optimo:
                row += f" {(length - self.optimo) / self.optimo * 100:>8.2f}"
            print(row)
        ganador = min(self.results, key=lambda a: self.results[a][0])
        print(f"\nMejor resultado: {ganador} ({self.results[ganador][0]:.1f})")
        print(f"{linea}\n")

    def run_all(self) -> None:
        self.run_ga()
        self.run_cbga()
        self.run_aco()
        self.print_resumen()


def main():
    ap = argparse.ArgumentParser(description="Comparativa GA vs ACO vs CBGA — Opción A")

    ap.add_argument("--instance",   required=True)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--no-2opt",    action="store_true")

    ap.add_argument("--pop",        type=int,   default=50)
    ap.add_argument("--pc",         type=float, default=0.9)
    ap.add_argument("--pm",         type=float, default=0.1)
    ap.add_argument("--elitism",    type=int,   default=5)
    ap.add_argument("--tournament", type=int,   default=3)

    ap.add_argument("--diversity",  type=float, default=0.15)
    ap.add_argument("--attempts",   type=int,   default=5)

    ap.add_argument("--ants",       type=int,   default=50)
    ap.add_argument("--alpha",      type=float, default=1.0)
    ap.add_argument("--beta",       type=float, default=5.0)
    ap.add_argument("--rho",        type=float, default=0.1)
    ap.add_argument("--q",          type=float, default=1.0)
    ap.add_argument("--tau0",       type=float, default=0.01)
    ap.add_argument("--no-elitist", action="store_true")

    args   = ap.parse_args()
    nombre = os.path.basename(args.instance)
    optimo = OPTIMOS.get(nombre)
    linea  = "=" * 56

    print(f"\n{linea}")
    print(f"Instancia  : {nombre}")
    print(f"Óptimo     : {optimo if optimo else 'desconocido'}")
    print(f"Presupuesto: {args.time_limit}s por algoritmo (Opción A)")
    print(f"Seed       : {args.seed}")
    print(f"{linea}")

    cities = read_tsp(args.instance)
    dist   = make_dist_matrix(cities)
    print(f"Ciudades   : {len(cities)}\n")

    comparador = Comparador(args=args, cities=cities, dist=dist, nombre=nombre)
    comparador.run_all()


if __name__ == "__main__":
    main()