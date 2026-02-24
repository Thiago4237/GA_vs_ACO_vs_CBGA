"""
visualizaciones.py — Gráficas del taller GA vs CBGA vs ACO.

Lee el CSV generado por runner.py y produce:
    1. Boxplot de longitudes
    2. Tabla resumen con GAP%
    3. Barras de tiempo promedio
    4. Curva de convergencia promedio (best-so-far vs tiempo)

Uso:
    python visualizaciones.py --csv resultados/resultados_berlin52.csv
"""

import csv
import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


COLORES = {"GA": "#4C72B0", "CBGA": "#DD8452", "ACO": "#55A868"}

# Óptimos conocidos para línea de referencia
OPTIMOS = {
    "berlin52.tsp": 7542,
    "eil51.tsp":    426,
    "att48.tsp":    10628,
    "st70.tsp":     675,
}


def leer_csv(path: str):
    datos      = {}
    historiales = {}
    nombre_inst = ""

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row["algoritmo"]
            nombre_inst = row["instancia"]
            if algo not in datos:
                datos[algo]       = {"lengths": [], "times": [], "gaps": []}
                historiales[algo] = []

            datos[algo]["lengths"].append(float(row["longitud"]))
            datos[algo]["times"].append(float(row["tiempo_s"]))
            if row.get("gap_pct") not in ("", None):
                datos[algo]["gaps"].append(float(row["gap_pct"]))

            if row.get("historial"):
                historiales[algo].append(json.loads(row["historial"]))

    return datos, historiales, nombre_inst


def stats(values):
    arr = np.array(values)
    return {"best": arr.min(), "worst": arr.max(),
            "mean": arr.mean(), "std": arr.std(), "median": np.median(arr)}


# ── 1. Boxplot ────────────────────────────────────────────────────────────────

def plot_boxplot(datos, titulo, outdir, nombre_inst):
    algoritmos = list(datos.keys())
    lengths    = [datos[a]["lengths"] for a in algoritmos]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(lengths, patch_artist=True, notch=False, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))

    for patch, algo in zip(bp["boxes"], algoritmos):
        patch.set_facecolor(COLORES.get(algo, "gray"))
        patch.set_alpha(0.75)

    for i, (vals, algo) in enumerate(zip(lengths, algoritmos), start=1):
        x = np.random.normal(i, 0.06, size=len(vals))
        ax.scatter(x, vals, alpha=0.4, s=18, color=COLORES.get(algo, "gray"), zorder=3)

    optimo = OPTIMOS.get(nombre_inst)
    if optimo:
        ax.axhline(optimo, color="red", linestyle="--", linewidth=1.2, label=f"Óptimo ({optimo})")
        ax.legend()

    ax.set_xticks(range(1, len(algoritmos) + 1))
    ax.set_xticklabels(algoritmos, fontsize=12)
    ax.set_ylabel("Longitud del tour", fontsize=11)
    ax.set_title(f"Distribución de resultados — {titulo}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(outdir, f"boxplot_{titulo}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Boxplot     → {path}")


# ── 2. Tabla resumen ──────────────────────────────────────────────────────────

def print_tabla(datos, nombre_inst):
    optimo = OPTIMOS.get(nombre_inst)
    print(f"\n{'='*70}")
    header = f"{'Algoritmo':<8} {'Mejor':>8} {'Peor':>8} {'Media':>10} {'Std':>8}"
    if optimo:
        header += f" {'GAP_best%':>10} {'GAP_med%':>10}"
    header += f" {'T.med(s)':>10}"
    print(header)
    print(f"{'-'*70}")
    for algo in datos:
        s = stats(datos[algo]["lengths"])
        t = stats(datos[algo]["times"])
        row = f"{algo:<8} {s['best']:>8.1f} {s['worst']:>8.1f} {s['mean']:>10.1f} {s['std']:>8.1f}"
        if optimo:
            row += f" {(s['best']-optimo)/optimo*100:>10.2f} {(s['mean']-optimo)/optimo*100:>10.2f}"
        row += f" {t['mean']:>10.2f}"
        print(row)
    print(f"{'='*70}\n")


def plot_tabla(datos, titulo, outdir, nombre_inst):
    optimo = OPTIMOS.get(nombre_inst)
    colnames = ["Algoritmo", "Mejor", "Peor", "Media", "Std"]
    if optimo:
        colnames += ["GAP_best%", "GAP_med%"]
    colnames.append("T.med(s)")

    filas = []
    for algo in datos:
        s = stats(datos[algo]["lengths"])
        t = stats(datos[algo]["times"])
        fila = [algo, f"{s['best']:.1f}", f"{s['worst']:.1f}",
                f"{s['mean']:.1f}", f"{s['std']:.1f}"]
        if optimo:
            fila += [f"{(s['best']-optimo)/optimo*100:.2f}",
                     f"{(s['mean']-optimo)/optimo*100:.2f}"]
        fila.append(f"{t['mean']:.2f}")
        filas.append(fila)

    fig, ax = plt.subplots(figsize=(11, 1.5 + 0.6 * len(filas)))
    ax.axis("off")
    tabla = ax.table(cellText=filas, colLabels=colnames, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(11)
    tabla.scale(1.2, 1.7)
    for j in range(len(colnames)):
        tabla[0, j].set_facecolor("#2C5F8A")
        tabla[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(filas) + 1):
        color = ["#EAF2FB", "#FFFFFF"][i % 2]
        for j in range(len(colnames)):
            tabla[i, j].set_facecolor(color)
    ax.set_title(f"Resumen estadístico — {titulo}", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(outdir, f"tabla_{titulo}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Tabla       → {path}")


# ── 3. Tiempos ────────────────────────────────────────────────────────────────

def plot_tiempos(datos, titulo, outdir):
    algoritmos = list(datos.keys())
    medias = [stats(datos[a]["times"])["mean"] for a in algoritmos]
    stds   = [stats(datos[a]["times"])["std"]  for a in algoritmos]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(algoritmos, medias, yerr=stds, capsize=5,
                  color=[COLORES.get(a, "gray") for a in algoritmos],
                  alpha=0.8, edgecolor="black")
    for bar, val in zip(bars, medias):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.1,
                f"{val:.2f}s", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Tiempo promedio (s)", fontsize=11)
    ax.set_title(f"Tiempo de ejecución — {titulo}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(outdir, f"tiempos_{titulo}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Tiempos     → {path}")


# ── 4. Curva de convergencia ──────────────────────────────────────────────────

def plot_convergencia(historiales, titulo, outdir, nombre_inst, time_limit=10.0):
    """
    Interpola cada historial en una grilla de tiempo común y promedia
    las curvas de todas las seeds → curva media con banda ±std.
    """
    t_grid = np.linspace(0, time_limit, 300)
    fig, ax = plt.subplots(figsize=(9, 5))

    for algo, hists in historiales.items():
        if not hists:
            continue

        curvas = []
        for hist in hists:
            if len(hist) < 2:
                continue
            ts  = np.array([p[0] for p in hist])
            ys  = np.array([p[1] for p in hist])
            # Interpolar en la grilla común (escalón — ffill)
            interp = np.interp(t_grid, ts, ys)
            curvas.append(interp)

        if not curvas:
            continue

        curvas = np.array(curvas)
        media  = curvas.mean(axis=0)
        std    = curvas.std(axis=0)

        color = COLORES.get(algo, "gray")
        ax.plot(t_grid, media, label=algo, color=color, linewidth=2)
        ax.fill_between(t_grid, media - std, media + std,
                        alpha=0.15, color=color)

    optimo = OPTIMOS.get(nombre_inst)
    if optimo:
        ax.axhline(optimo, color="red", linestyle="--", linewidth=1.2,
                   label=f"Óptimo ({optimo})")

    ax.set_xlabel("Tiempo (s)", fontsize=11)
    ax.set_ylabel("Longitud best-so-far", fontsize=11)
    ax.set_title(f"Convergencia promedio (±std) — {titulo}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(outdir, f"convergencia_{titulo}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Convergencia → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",        required=True)
    ap.add_argument("--outdir",     default="graficos")
    ap.add_argument("--time-limit", type=float, default=10.0,
                    help="Tiempo límite usado en el runner (para eje X de convergencia)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nLeyendo {args.csv}...")
    datos, historiales, nombre_inst = leer_csv(args.csv)

    algoritmos = list(datos.keys())
    n_seeds    = len(datos[algoritmos[0]]["lengths"])
    titulo     = os.path.splitext(nombre_inst)[0]

    print(f"Algoritmos  : {algoritmos}")
    print(f"Seeds       : {n_seeds}")
    print(f"Instancia   : {nombre_inst}")

    print_tabla(datos, nombre_inst)

    plot_boxplot(datos, titulo, args.outdir, nombre_inst)
    plot_tabla(datos, titulo, args.outdir, nombre_inst)
    plot_tiempos(datos, titulo, args.outdir)
    plot_convergencia(historiales, titulo, args.outdir, nombre_inst, args.time_limit)

    print("\nTodas las gráficas generadas.\n")


if __name__ == "__main__":
    main()