"""
runner.py — Experimento de 30 seeds. Opción A (tiempo fijo).
Guarda resultados + historial de convergencia + GAP%.
"""

import time
import argparse
import os
import csv
import json
from typing import List, Dict

from utils    import read_tsp, make_dist_matrix, tour_length, tour_length_idx
from GA_tsp   import genetic_tsp
from ACO_tsp  import ant_colony_tsp
from CBGA_tsp import cbga_tsp

# Óptimos conocidos — agregar los tuyos aquí
OPTIMOS = {
    "berlin52.tsp": 7542,
    "eil51.tsp":    426,
    "att48.tsp":    10628,
    "st70.tsp":     675,
}


def stats(values: List[float]) -> Dict:
    n    = len(values)
    mean = sum(values) / n
    std  = (sum((x - mean) ** 2 for x in values) / n) ** 0.5
    return {"best": min(values), "worst": max(values), "mean": mean, "std": std}


def gap(length: float, optimo: float) -> float:
    return (length - optimo) / optimo * 100


def run_ga(cities, seed, args):
    t0 = time.perf_counter()
    tour, historial = genetic_tsp(
        cities=cities, pop_size=args.pop, time_limit=args.time_limit,
        pc=args.pc, pm=args.pm, elitism_k=args.elitism,
        tournament_k=args.tournament, local_2opt=not args.no_2opt, seed=seed,
    )
    return tour_length(tour), time.perf_counter() - t0, historial


def run_cbga(cities, seed, args):
    t0 = time.perf_counter()
    tour, historial = cbga_tsp(
        cities=cities, pop_size=args.pop, time_limit=args.time_limit,
        pc=args.pc, pm=args.pm, tournament_k=args.tournament,
        min_diversity=args.diversity, attempts_per_gen=args.attempts,
        local_2opt=not args.no_2opt, seed=seed,
    )
    return tour_length(tour), time.perf_counter() - t0, historial


def run_aco(cities, dist, seed, args):
    t0 = time.perf_counter()
    trail, historial = ant_colony_tsp(
        dist=dist, num_ants=args.ants, time_limit=args.time_limit,
        alpha=args.alpha, beta=args.beta, rho=args.rho, Q=args.q,
        tau0=args.tau0, elitist=not args.no_elitist,
        local_2opt=not args.no_2opt, seed=seed,
    )
    return tour_length_idx(trail, dist), time.perf_counter() - t0, historial


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

    nombre = os.path.basename(args.instance)
    seeds  = list(range(args.seed_start, args.seed_start + args.seeds))
    optimo = OPTIMOS.get(nombre)
    t_est  = args.seeds * 3 * args.time_limit

    print(f"\n{'='*62}")
    print(f"Instancia  : {nombre}")
    print(f"Óptimo     : {optimo if optimo else 'desconocido'}")
    print(f"Seeds      : {args.seed_start} → {args.seed_start + args.seeds - 1}")
    print(f"Presupuesto: {args.time_limit}s por algoritmo (Opción A)")
    print(f"Tiempo estimado: ~{t_est:.0f}s ({t_est/60:.1f} min)")
    print(f"{'='*62}\n")

    cities = read_tsp(args.instance)
    dist   = make_dist_matrix(cities)

    registros  = []
    historiales = {a: [] for a in ["GA", "CBGA", "ACO"]}
    resultados  = {a: {"lengths": [], "times": []} for a in ["GA", "CBGA", "ACO"]}

    for i, seed in enumerate(seeds, 1):
        print(f"  Seed {seed:3d} ({i:2d}/{args.seeds})", end="  ")

        len_ga,   t_ga,   hist_ga   = run_ga(cities, seed, args)
        len_cbga, t_cbga, hist_cbga = run_cbga(cities, seed, args)
        len_aco,  t_aco,  hist_aco  = run_aco(cities, dist, seed, args)

        # GAP%
        if optimo:
            g_ga   = f"{gap(len_ga,   optimo):.2f}%"
            g_cbga = f"{gap(len_cbga, optimo):.2f}%"
            g_aco  = f"{gap(len_aco,  optimo):.2f}%"
            print(f"GA={len_ga:.0f}({g_ga})  CBGA={len_cbga:.0f}({g_cbga})  ACO={len_aco:.0f}({g_aco})")
        else:
            print(f"GA={len_ga:.0f}  CBGA={len_cbga:.0f}  ACO={len_aco:.0f}")

        for algo, length, t, hist in [
            ("GA",   len_ga,   t_ga,   hist_ga),
            ("CBGA", len_cbga, t_cbga, hist_cbga),
            ("ACO",  len_aco,  t_aco,  hist_aco),
        ]:
            resultados[algo]["lengths"].append(length)
            resultados[algo]["times"].append(t)
            historiales[algo].append(hist)
            registros.append({
                "instancia":  nombre,
                "algoritmo":  algo,
                "seed":       seed,
                "longitud":   round(length, 2),
                "tiempo_s":   round(t, 4),
                "gap_pct":    round(gap(length, optimo), 4) if optimo else "",
                "historial":  json.dumps(hist),   # guardado como JSON en el CSV
            })

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = args.output
    if csv_path is None:
        base = os.path.splitext(nombre)[0]
        os.makedirs("resultados", exist_ok=True)
        csv_path = os.path.join("resultados", f"resultados_{base}.csv")

    campos = ["instancia", "algoritmo", "seed", "longitud", "tiempo_s", "gap_pct", "historial"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(registros)

    print(f"\n  CSV guardado → {csv_path}")

    # ── Resumen ───────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"RESUMEN — {nombre}  ({args.seeds} seeds × {args.time_limit}s)")
    print(f"{'='*62}")
    header = f"{'Algoritmo':<8} {'Mejor':>8} {'Peor':>8} {'Media':>10} {'Std':>8}"
    if optimo:
        header += f" {'GAP_best%':>10} {'GAP_med%':>10}"
    header += f" {'T.med(s)':>10}"
    print(header)
    print(f"{'-'*62}")

    for algo in ["GA", "CBGA", "ACO"]:
        s = stats(resultados[algo]["lengths"])
        t = stats(resultados[algo]["times"])
        row = f"{algo:<8} {s['best']:>8.1f} {s['worst']:>8.1f} {s['mean']:>10.1f} {s['std']:>8.1f}"
        if optimo:
            row += f" {gap(s['best'], optimo):>10.2f} {gap(s['mean'], optimo):>10.2f}"
        row += f" {t['mean']:>10.2f}"
        print(row)

    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()