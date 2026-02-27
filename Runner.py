"""
Runner.py — Experimento de 30 seeds. Opción A (tiempo fijo).
Guarda resultados + historial de convergencia + GAP%.
Implementación orientada a objetos (OOP).
"""

import time
import argparse
import os
import csv
import json
from typing import List, Dict, Tuple

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


def _stats(values: List[float]) -> Dict:
    n    = len(values)
    mean = sum(values) / n
    std  = (sum((x - mean) ** 2 for x in values) / n) ** 0.5
    return {"best": min(values), "worst": max(values), "mean": mean, "std": std}


def _gap(length: float, optimo: float) -> float:
    return (length - optimo) / optimo * 100


class ExperimentRunner:
    """
    Ejecuta el experimento multi-seed y guarda resultados en CSV.

    Parámetros
    ----------
    args    : namespace de argparse
    cities  : lista de ciudades
    dist    : matriz de distancias
    nombre  : nombre del archivo de instancia
    seeds   : lista de seeds a ejecutar
    """

    def __init__(self, args, cities, dist, nombre: str, seeds: List[int]):
        self.args    = args
        self.cities  = cities
        self.dist    = dist
        self.nombre  = nombre
        self.seeds   = seeds
        self.optimo  = OPTIMOS.get(nombre)

        self.registros:   List[Dict]         = []
        self.resultados:  Dict               = {a: {"lengths": [], "times": []} for a in ["GA", "CBGA", "ACO"]}

    # ── Ejecución por algoritmo ───────────────────────────────────────────

    def _run_ga(self, seed: int) -> Tuple[float, float, list]:
        algo = GeneticAlgorithm(
            cities=self.cities, pop_size=self.args.pop, time_limit=self.args.time_limit,
            pc=self.args.pc, pm=self.args.pm, elitism_k=self.args.elitism,
            tournament_k=self.args.tournament, local_2opt=not self.args.no_2opt, seed=seed,
        )
        t0 = time.perf_counter()
        tour, historial = algo.run()
        return tour_length(tour), time.perf_counter() - t0, historial

    def _run_cbga(self, seed: int) -> Tuple[float, float, list]:
        algo = ChuBeasleyGA(
            cities=self.cities, pop_size=self.args.pop, time_limit=self.args.time_limit,
            pc=self.args.pc, pm=self.args.pm, tournament_k=self.args.tournament,
            min_diversity=self.args.diversity, attempts_per_gen=self.args.attempts,
            local_2opt=not self.args.no_2opt, seed=seed,
        )
        t0 = time.perf_counter()
        tour, historial = algo.run()
        return tour_length(tour), time.perf_counter() - t0, historial

    def _run_aco(self, seed: int) -> Tuple[float, float, list]:
        algo = AntColony(
            dist=self.dist, num_ants=self.args.ants, time_limit=self.args.time_limit,
            alpha=self.args.alpha, beta=self.args.beta, rho=self.args.rho, Q=self.args.q,
            tau0=self.args.tau0, elitist=not self.args.no_elitist,
            local_2opt=not self.args.no_2opt, seed=seed,
        )
        t0 = time.perf_counter()
        trail, historial = algo.run()
        return tour_length_idx(trail, self.dist), time.perf_counter() - t0, historial

    # ── Registro de resultados ────────────────────────────────────────────

    def _registrar(self, algo: str, length: float, t: float, hist: list) -> None:
        self.resultados[algo]["lengths"].append(length)
        self.resultados[algo]["times"].append(t)
        self.registros.append({
            "instancia": self.nombre,
            "algoritmo": algo,
            "seed":      self.seeds[len(self.resultados[algo]["lengths"]) - 1],
            "longitud":  round(length, 2),
            "tiempo_s":  round(t, 4),
            "gap_pct":   round(_gap(length, self.optimo), 4) if self.optimo else "",
            "historial": json.dumps(hist),
        })

    # ── Ejecución completa ────────────────────────────────────────────────

    def run(self) -> None:
        for i, seed in enumerate(self.seeds, 1):
            print(f"  Seed {seed:3d} ({i:2d}/{len(self.seeds)})", end="  ")

            len_ga,   t_ga,   hist_ga   = self._run_ga(seed)
            len_cbga, t_cbga, hist_cbga = self._run_cbga(seed)
            len_aco,  t_aco,  hist_aco  = self._run_aco(seed)

            if self.optimo:
                print(f"GA={len_ga:.0f}({_gap(len_ga, self.optimo):.2f}%)  "
                      f"CBGA={len_cbga:.0f}({_gap(len_cbga, self.optimo):.2f}%)  "
                      f"ACO={len_aco:.0f}({_gap(len_aco, self.optimo):.2f}%)")
            else:
                print(f"GA={len_ga:.0f}  CBGA={len_cbga:.0f}  ACO={len_aco:.0f}")

            self._registrar("GA",   len_ga,   t_ga,   hist_ga)
            self._registrar("CBGA", len_cbga, t_cbga, hist_cbga)
            self._registrar("ACO",  len_aco,  t_aco,  hist_aco)

    # ── CSV ───────────────────────────────────────────────────────────────

    def save_csv(self, csv_path: str) -> None:
        campos = ["instancia", "algoritmo", "seed", "longitud", "tiempo_s", "gap_pct", "historial"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(self.registros)
        print(f"\n  CSV guardado → {csv_path}")

    # ── Resumen en consola ────────────────────────────────────────────────

    def print_resumen(self) -> None:
        n_seeds = len(self.seeds)
        print(f"\n{'='*62}")
        print(f"RESUMEN — {self.nombre}  ({n_seeds} seeds × {self.args.time_limit}s)")
        print(f"{'='*62}")
        header = f"{'Algoritmo':<8} {'Mejor':>8} {'Peor':>8} {'Media':>10} {'Std':>8}"
        if self.optimo:
            header += f" {'GAP_best%':>10} {'GAP_med%':>10}"
        header += f" {'T.med(s)':>10}"
        print(header)
        print("-" * 62)
        for algo in ["GA", "CBGA", "ACO"]:
            s = _stats(self.resultados[algo]["lengths"])
            t = _stats(self.resultados[algo]["times"])
            row = f"{algo:<8} {s['best']:>8.1f} {s['worst']:>8.1f} {s['mean']:>10.1f} {s['std']:>8.1f}"
            if self.optimo:
                row += f" {_gap(s['best'], self.optimo):>10.2f} {_gap(s['mean'], self.optimo):>10.2f}"
            row += f" {t['mean']:>10.2f}"
            print(row)
        print(f"{'='*62}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance",   required=True)
    ap.add_argument("--seeds",      type=int,   default=30)
    ap.add_argument("--seed-start", type=int,   default=1)
    ap.add_argument("--output",     type=str,   default=None)
    ap.add_argument("--time-limit", type=float, default=10.0)
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
    args = ap.parse_args()

    nombre  = os.path.basename(args.instance)
    seeds   = list(range(args.seed_start, args.seed_start + args.seeds))
    optimo  = OPTIMOS.get(nombre)
    t_est   = args.seeds * 3 * args.time_limit

    print(f"\n{'='*62}")
    print(f"Instancia  : {nombre}")
    print(f"Óptimo     : {optimo if optimo else 'desconocido'}")
    print(f"Seeds      : {args.seed_start} → {args.seed_start + args.seeds - 1}")
    print(f"Presupuesto: {args.time_limit}s por algoritmo (Opción A)")
    print(f"Tiempo estimado: ~{t_est:.0f}s ({t_est/60:.1f} min)")
    print(f"{'='*62}\n")

    cities = read_tsp(args.instance)
    dist   = make_dist_matrix(cities)

    runner = ExperimentRunner(args=args, cities=cities, dist=dist, nombre=nombre, seeds=seeds)
    runner.run()

    csv_path = args.output
    if csv_path is None:
        base = os.path.splitext(nombre)[0]
        os.makedirs("resultados", exist_ok=True)
        csv_path = os.path.join("resultados", f"resultados_{base}.csv")

    runner.save_csv(csv_path)
    runner.print_resumen()


if __name__ == "__main__":
    main()