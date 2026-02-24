# GA vs ACO vs CBGA — TSP Benchmark

Comparativo experimental de tres algoritmos metaheurísticos aplicados al **Problema del Viajante (TSP)** usando instancias estándar de TSPLIB.

| Algoritmo | Tipo | Descripción |
|-----------|------|-------------|
| **GA** | Genético clásico | Selección por torneo, cruce OX, mutación swap, elitismo |
| **ACO** | Colonia de hormigas | Feromonas + heurística η, depósito elitista |
| **CBGA** | Chu-Beasley GA | Control de diversidad, reemplazo selectivo, sin duplicados |

Todos los algoritmos usan **2-opt local search** y compiten bajo **presupuesto por tiempo fijo** (mismo T segundos por algoritmo y seed), garantizando una comparación justa.

---

## ¿Qué es 2-opt?

**2-opt** es una búsqueda local que mejora un tour eliminando cruces entre rutas. En cada paso, toma dos aristas del tour, las elimina y reconecta los segmentos en el orden inverso. Si el nuevo tour es más corto, lo acepta y repite el proceso hasta que ningún intercambio mejore la solución.

```
Antes:  A──►B        Después: A──►C
            │                      │
            ▼                      ▼
        D◄──C              D◄──B
```

En este proyecto se aplica **al mejor individuo/hormiga de cada ciclo**, no a todos, para mantener el tiempo dentro del presupuesto sin sacrificar calidad.

---

## Estructura del proyecto

```
proyecto/
├── data/                   # Instancias TSPLIB (.tsp)
│   ├── berlin52.tsp
│   ├── eil51.tsp
│   ├── att48.tsp
│   └── st70.tsp
├── resultados/             # CSVs generados por Runner.py
├── graficos/               # Imágenes generadas por visualizaciones.py
├── utils.py                # Lectura de .tsp y evaluación de tours
├── GA_tsp.py               # Implementación GA
├── ACO_tsp.py              # Implementación ACO
├── CBGA_tsp.py             # Implementación CBGA
├── main.py                 # Ejecución de una sola seed
├── Runner.py               # Experimento de 30 seeds
└── visualizaciones.py      # Gráficas desde el CSV
```

---

## Ejecución

### Una seed (prueba rápida)
```bash
python main.py --instance data/berlin52.tsp
```

### 30 seeds (experimento completo)
```bash
python Runner.py --instance data/berlin52.tsp
```
→ Genera `resultados/resultados_berlin52.csv`

### Visualizaciones
```bash
python visualizaciones.py --csv resultados/resultados_berlin52.csv
```
→ Genera boxplot, tabla, tiempos y curva de convergencia en `graficos/`

---

## Parámetros disponibles

Todos los scripts aceptan los siguientes argumentos. Los valores mostrados son los **defaults**.

### Presupuesto (aplica a todos los algoritmos)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--instance` | *(requerido)* | Ruta al archivo `.tsp` |
| `--time-limit` | `10.0` | Segundos de ejecución por algoritmo |
| `--seed` | `42` | Semilla (solo en `main.py`) |
| `--no-2opt` | `False` | Desactiva 2-opt en todos los algoritmos |

### GA y CBGA

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--pop` | `50` | Tamaño de la población |
| `--pc` | `0.9` | Probabilidad de cruce |
| `--pm` | `0.1` | Probabilidad de mutación |
| `--elitism` | `5` | Individuos élite por generación *(solo GA)* |
| `--tournament` | `3` | Tamaño del torneo de selección |
| `--diversity` | `0.15` | Diversidad mínima entre tours *(solo CBGA)* |
| `--attempts` | `5` | Intentos de reproducción por ciclo *(solo CBGA)* |

### ACO

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--ants` | `50` | Número de hormigas por iteración |
| `--alpha` | `1.0` | Peso de la feromona (τ) |
| `--beta` | `5.0` | Peso de la heurística (η = 1/d) |
| `--rho` | `0.1` | Tasa de evaporación |
| `--q` | `1.0` | Constante de depósito |
| `--tau0` | `0.01` | Feromona inicial en todas las aristas |
| `--no-elitist` | `False` | Depósito por todas las hormigas (no solo la mejor) |

### Runner

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--seeds` | `30` | Número de seeds a ejecutar |
| `--seed-start` | `1` | Primera seed del rango |
| `--output` | `resultados/resultados_<instancia>.csv` | Ruta del CSV de salida |

---

## Ejemplo con parámetros personalizados

```bash
# Prueba con 5s por algoritmo y población de 100
python main.py --instance data/eil51.tsp --time-limit 5 --pop 100 --pm 0.05

# Runner con 10 seeds y ACO más agresivo
python Runner.py --instance data/berlin52.tsp --seeds 10 --beta 8 --rho 0.3

# Comparar sin búsqueda local
python main.py --instance data/berlin52.tsp --no-2opt
```

---

## Óptimos conocidos (TSPLIB)

| Instancia | Ciudades | Óptimo |
|-----------|----------|--------|
| `att48.tsp` | 48 | 10628 |
| `eil51.tsp` | 51 | 426 |
| `berlin52.tsp` | 52 | 7542 |
| `st70.tsp` | 70 | 675 |