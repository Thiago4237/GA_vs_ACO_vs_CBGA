"""
visualizaciones.py — Gráficas del taller GA vs CBGA vs ACO.
Implementación orientada a objetos (OOP).

Lee el CSV generado por Runner.py y produce:
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

OPTIMOS = {
    "berlin52.tsp": 7542,
    "eil51.tsp":    426,
    "att48.tsp":    10628,
    "st70.tsp":     675,
}


class Visualizador:
    """
    Genera todas las gráficas del experimento a partir de un CSV.

    Parámetros
    ----------
    csv_path   : ruta al CSV generado por Runner.py
    outdir     : carpeta de salida para las imágenes
    time_limit : presupuesto de tiempo usado (para eje X de convergencia)
    """

    def __init__(self, csv_path: str, outdir: str = "graficos", time_limit: float = 10.0):
        self.csv_path   = csv_path
        self.outdir     = outdir
        self.time_limit = time_limit

        self.datos:       Dict = {}
        self.historiales: Dict = {}
        self.nombre_inst: str  = ""
        self.titulo:      str  = ""

        os.makedirs(outdir, exist_ok=True)

    # ── Lectura de datos ──────────────────────────────────────────────────

    def cargar(self) -> None:
        """Lee el CSV y carga datos e historiales en memoria."""
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                algo = row["algoritmo"]
                self.nombre_inst = row["instancia"]
                if algo not in self.datos:
                    self.datos[algo]       = {"lengths": [], "times": [], "gaps": []}
                    self.historiales[algo] = []

                self.datos[algo]["lengths"].append(float(row["longitud"]))
                self.datos[algo]["times"].append(float(row["tiempo_s"]))
                if row.get("gap_pct") not in ("", None):
                    self.datos[algo]["gaps"].append(float(row["gap_pct"]))
                if row.get("historial"):
                    self.historiales[algo].append(json.loads(row["historial"]))

        self.titulo  = os.path.splitext(self.nombre_inst)[0]
        algoritmos   = list(self.datos.keys())
        n_seeds      = len(self.datos[algoritmos[0]]["lengths"])
        print(f"Algoritmos : {algoritmos}")
        print(f"Seeds      : {n_seeds}")
        print(f"Instancia  : {self.nombre_inst}")

    # ── Estadísticas ──────────────────────────────────────────────────────

    @staticmethod
    def _stats(values: List[float]) -> Dict:
        arr = np.array(values)
        return {
            "best": arr.min(), "worst": arr.max(),
            "mean": arr.mean(), "std":  arr.std(),
        }

    # ── 1. Boxplot ────────────────────────────────────────────────────────

    def plot_boxplot(self) -> None:
        algoritmos = list(self.datos.keys())
        lengths    = [self.datos[a]["lengths"] for a in algoritmos]
        optimo     = OPTIMOS.get(self.nombre_inst)

        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(lengths, patch_artist=True, notch=False, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))

        for patch, algo in zip(bp["boxes"], algoritmos):
            patch.set_facecolor(COLORES.get(algo, "gray"))
            patch.set_alpha(0.75)

        for i, (vals, algo) in enumerate(zip(lengths, algoritmos), start=1):
            x = np.random.normal(i, 0.06, size=len(vals))
            ax.scatter(x, vals, alpha=0.4, s=18, color=COLORES.get(algo, "gray"), zorder=3)

        if optimo:
            ax.axhline(optimo, color="red", linestyle="--", linewidth=1.2,
                       label=f"Óptimo ({optimo})")
            ax.legend()

        ax.set_xticks(range(1, len(algoritmos) + 1))
        ax.set_xticklabels(algoritmos, fontsize=12)
        ax.set_ylabel("Longitud del tour", fontsize=11)
        ax.set_title(f"Distribución de resultados — {self.titulo}",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(self.outdir, f"boxplot_{self.titulo}.png")
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"  Boxplot      → {path}")

    # ── 2. Tabla resumen ──────────────────────────────────────────────────

    def print_tabla(self) -> None:
        optimo = OPTIMOS.get(self.nombre_inst)
        print(f"\n{'='*70}")
        header = f"{'Algoritmo':<8} {'Mejor':>8} {'Peor':>8} {'Media':>10} {'Std':>8}"
        if optimo:
            header += f" {'GAP_best%':>10} {'GAP_med%':>10}"
        header += f" {'T.med(s)':>10}"
        print(header)
        print("-" * 70)
        for algo in self.datos:
            s = self._stats(self.datos[algo]["lengths"])
            t = self._stats(self.datos[algo]["times"])
            row = f"{algo:<8} {s['best']:>8.1f} {s['worst']:>8.1f} {s['mean']:>10.1f} {s['std']:>8.1f}"
            if optimo:
                row += (f" {(s['best']-optimo)/optimo*100:>10.2f}"
                        f" {(s['mean']-optimo)/optimo*100:>10.2f}")
            row += f" {t['mean']:>10.2f}"
            print(row)
        print(f"{'='*70}\n")

    def plot_tabla(self) -> None:
        optimo   = OPTIMOS.get(self.nombre_inst)
        colnames = ["Algoritmo", "Mejor", "Peor", "Media", "Std"]
        if optimo:
            colnames += ["GAP_best%", "GAP_med%"]
        colnames.append("T.med(s)")

        filas = []
        for algo in self.datos:
            s = self._stats(self.datos[algo]["lengths"])
            t = self._stats(self.datos[algo]["times"])
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
        ax.set_title(f"Resumen estadístico — {self.titulo}",
                     fontsize=12, fontweight="bold", pad=20)
        plt.tight_layout()
        path = os.path.join(self.outdir, f"tabla_{self.titulo}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Tabla        → {path}")

    # ── 3. Tiempos ────────────────────────────────────────────────────────

    def plot_tiempos(self) -> None:
        algoritmos = list(self.datos.keys())
        medias = [self._stats(self.datos[a]["times"])["mean"] for a in algoritmos]
        stds   = [self._stats(self.datos[a]["times"])["std"]  for a in algoritmos]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(algoritmos, medias, yerr=stds, capsize=5,
                      color=[COLORES.get(a, "gray") for a in algoritmos],
                      alpha=0.8, edgecolor="black")
        for bar, val in zip(bars, medias):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.1,
                    f"{val:.2f}s", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Tiempo promedio (s)", fontsize=11)
        ax.set_title(f"Tiempo de ejecución — {self.titulo}",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(self.outdir, f"tiempos_{self.titulo}.png")
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"  Tiempos      → {path}")

    # ── 4. Convergencia ───────────────────────────────────────────────────

    def plot_convergencia(self) -> None:
        """Curva promedio (±std) de best-so-far vs tiempo para cada algoritmo."""
        t_grid = np.linspace(0, self.time_limit, 300)
        optimo = OPTIMOS.get(self.nombre_inst)

        fig, ax = plt.subplots(figsize=(9, 5))
        for algo, hists in self.historiales.items():
            curvas = []
            for hist in hists:
                if len(hist) < 2:
                    continue
                ts = np.array([p[0] for p in hist])
                ys = np.array([p[1] for p in hist])
                curvas.append(np.interp(t_grid, ts, ys))

            if not curvas:
                continue

            curvas = np.array(curvas)
            media  = curvas.mean(axis=0)
            std    = curvas.std(axis=0)
            color  = COLORES.get(algo, "gray")
            ax.plot(t_grid, media, label=algo, color=color, linewidth=2)
            ax.fill_between(t_grid, media - std, media + std, alpha=0.15, color=color)

        if optimo:
            ax.axhline(optimo, color="red", linestyle="--", linewidth=1.2,
                       label=f"Óptimo ({optimo})")

        ax.set_xlabel("Tiempo (s)", fontsize=11)
        ax.set_ylabel("Longitud best-so-far", fontsize=11)
        ax.set_title(f"Convergencia promedio (±std) — {self.titulo}",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        path = os.path.join(self.outdir, f"convergencia_{self.titulo}.png")
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"  Convergencia → {path}")

    # ── Generar todo ──────────────────────────────────────────────────────

    def generar_todo(self) -> None:
        self.print_tabla()
        self.plot_boxplot()
        self.plot_tabla()
        self.plot_tiempos()
        self.plot_convergencia()
        print("\nTodas las gráficas generadas.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",        required=True)
    ap.add_argument("--outdir",     default="graficos")
    ap.add_argument("--time-limit", type=float, default=10.0)
    args = ap.parse_args()

    print(f"\nLeyendo {args.csv}...")
    viz = Visualizador(csv_path=args.csv, outdir=args.outdir, time_limit=args.time_limit)
    viz.cargar()
    viz.generar_todo()


if __name__ == "__main__":
    main()