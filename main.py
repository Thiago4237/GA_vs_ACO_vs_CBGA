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
from GA_tsp   import genetic_tsp
from ACO_tsp  import ant_colony_tsp
from CBGA_tsp import cbga_tsp

OPTIMOS = {
    "berlin52.tsp": 7542,
    "eil51.tsp":    426,
    "att48.tsp":    10628,
    "st70.tsp":     675,
}


def main():
    ap = argparse.ArgumentParser(description="Comparativa GA vs ACO vs CBGA — Opción A")

    ap.add_argument("--instance",   required=True)
    ap.add_argument("--time-limit", type=float, default=10.0,
                    help="Segundos por algoritmo (igual para todos)")
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

    args = ap.parse_args()

    nombre   = os.path.basename(args.instance)
    use_2opt = not args.no_2opt
    optimo   = OPTIMOS.get(nombre)
    linea    = "=" * 56

    print(f"\n{linea}")
    print(f"Instancia  : {nombre}")
    print(f"Óptimo     : {optimo if optimo else 'desconocido'}")
    print(f"Presupuesto: {args.time_limit}s por algoritmo (Opción A)")
    print(f"Seed       : {args.seed}")
    print(f"{linea}")

    cities = read_tsp(args.instance)
    dist   = make_dist_matrix(cities)
    print(f"Ciudades   : {len(cities)}\n")

    results = {}

    def mostrar(algo, length, t):
        gap_str = ""
        if optimo:
            gap_str = f"  GAP={((length - optimo) / optimo * 100):.2f}%"
        print(f"Resultado  : {length:.1f}  ({t:.2f}s){gap_str}\n")

    # ── GA ────────────────────────────────────────────────────────────────
    print(f"--- GA ---")
    print(f"pop={args.pop}  pc={args.pc}  pm={args.pm}  élite={args.elitism}  torneo={args.tournament}  2opt={use_2opt}")
    t0 = time.perf_counter()
    ga_tour, _ = genetic_tsp(          # ← desempacar tupla
        cities=cities, pop_size=args.pop, time_limit=args.time_limit,
        pc=args.pc, pm=args.pm, elitism_k=args.elitism,
        tournament_k=args.tournament, local_2opt=use_2opt, seed=args.seed,
    )
    ga_time = time.perf_counter() - t0
    ga_len  = tour_length(ga_tour)
    results["GA"] = (ga_len, ga_time)
    mostrar("GA", ga_len, ga_time)

    # ── CBGA ──────────────────────────────────────────────────────────────
    print(f"--- CBGA ---")
    print(f"pop={args.pop}  pc={args.pc}  pm={args.pm}  torneo={args.tournament}  div={args.diversity}  intentos={args.attempts}  2opt={use_2opt}")
    t0 = time.perf_counter()
    cbga_tour, _ = cbga_tsp(           # ← desempacar tupla
        cities=cities, pop_size=args.pop, time_limit=args.time_limit,
        pc=args.pc, pm=args.pm, tournament_k=args.tournament,
        min_diversity=args.diversity, attempts_per_gen=args.attempts,
        local_2opt=use_2opt, seed=args.seed,
    )
    cbga_time = time.perf_counter() - t0
    cbga_len  = tour_length(cbga_tour)
    results["CBGA"] = (cbga_len, cbga_time)
    mostrar("CBGA", cbga_len, cbga_time)

    # ── ACO ───────────────────────────────────────────────────────────────
    print(f"--- ACO ---")
    print(f"ants={args.ants}  α={args.alpha}  β={args.beta}  ρ={args.rho}  Q={args.q}  elitist={not args.no_elitist}  2opt={use_2opt}")
    t0 = time.perf_counter()
    aco_trail, _ = ant_colony_tsp(     # ← desempacar tupla
        dist=dist, num_ants=args.ants, time_limit=args.time_limit,
        alpha=args.alpha, beta=args.beta, rho=args.rho, Q=args.q,
        tau0=args.tau0, elitist=not args.no_elitist,
        local_2opt=use_2opt, seed=args.seed,
    )
    aco_time = time.perf_counter() - t0
    aco_len  = tour_length_idx(aco_trail, dist)
    results["ACO"] = (aco_len, aco_time)
    mostrar("ACO", aco_len, aco_time)

    # ── Resumen ───────────────────────────────────────────────────────────
    print(f"{linea}")
    header = f"{'Algoritmo':<10} {'Longitud':>12} {'Tiempo (s)':>12}"
    if optimo:
        header += f" {'GAP%':>8}"
    print(header)
    print(f"{'-'*46}")
    for algo, (length, t) in results.items():
        row = f"{algo:<10} {length:>12.1f} {t:>12.2f}"
        if optimo:
            row += f" {(length - optimo) / optimo * 100:>8.2f}"
        print(row)

    ganador = min(results, key=lambda a: results[a][0])
    print(f"\nMejor resultado: {ganador} ({results[ganador][0]:.1f})")
    print(f"{linea}\n")


if __name__ == "__main__":
    main()