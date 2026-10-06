# -*- coding: utf-8 -*-
"""
Taller 5 - Punto 4: Exploración con Python (EDA)
Análisis de tráfico LAN y predicción bajo congestión
Universidad Sergio Arboleda - Maestría en Inteligencia Artificial
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de Rutas Dinámicas y Multiplataforma
def resolver_ruta_csv(nombre_archivo="ventanas.csv") -> Path:
    """
    Localiza dinámicamente la ruta del archivo CSV para garantizar portabilidad
    total en cualquier sistema operativo (Windows, Linux, macOS) y desde cualquier
    directorio de ejecución (terminal, VS Code, PyCharm, etc.), con soporte para
    argumentos de línea de comandos.
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", type=str, default=None, help="Ruta personalizada al archivo CSV")
    args, unknown = parser.parse_known_args()

    # 1. Prioridad: argumento explícito --csv
    if args.csv and Path(args.csv).exists():
        return Path(args.csv).resolve()

    # 2. Argumento posicional que termine en .csv
    for arg in unknown:
        if arg.lower().endswith(".csv") and Path(arg).exists():
            return Path(arg).resolve()

    # 3. Variable de entorno opcional
    if "VENTANAS_CSV_PATH" in os.environ and Path(os.environ["VENTANAS_CSV_PATH"]).exists():
        return Path(os.environ["VENTANAS_CSV_PATH"]).resolve()

    # 4. Búsqueda contextual relativa al script o al directorio de trabajo
    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidatos = [
        script_dir.parent / nombre_archivo,              # Raíz del proyecto (si se ejecuta desde scripts/)
        script_dir / nombre_archivo,                     # Mismo directorio del script
        script_dir.parent.parent / nombre_archivo,       # Dos niveles arriba
        Path.cwd() / nombre_archivo,                     # Directorio de trabajo actual
        Path.cwd() / "scripts" / nombre_archivo,
        Path.cwd() / "data" / nombre_archivo,
        Path.cwd() / "dataset" / nombre_archivo,
        Path.cwd().parent / nombre_archivo,
    ]

    for c in candidatos:
        if c.exists():
            return c.resolve()

    # 5. Búsqueda recursiva en el árbol de directorios si el archivo fue movido
    for base in [script_dir.parent, Path.cwd()]:
        try:
            encontrados = list(base.glob(f"**/{nombre_archivo}"))
            if encontrados:
                return encontrados[0].resolve()
        except Exception:
            pass

    # Fallback por defecto
    return (script_dir.parent / nombre_archivo).resolve()

CSV_PATH = resolver_ruta_csv("ventanas.csv")
BASE_DIR = CSV_PATH.parent
GRAFICOS_DIR = BASE_DIR / "graficos"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"

# 2. Cargar datos
print("=" * 70)
print("1. CARGA E INSPECCIÓN DE DATOS (EDA)")
print(f"Ruta base detectada:   {BASE_DIR}")
print(f"Dataset localizado:    {CSV_PATH}")
print(f"Directorio de gráficos:{GRAFICOS_DIR}")
print("=" * 70)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo '{CSV_PATH.name}'.\n"
        f"Ruta intentada: {CSV_PATH}\n"
        f"Sugerencia: Ejecute primero 'python scripts/construir_dataset.py' o indique la ruta con:\n"
        f"python scripts/exploracion_eda.py --csv /ruta/a/{CSV_PATH.name}"
    )

df = pd.read_csv(CSV_PATH, parse_dates=["inicio"], encoding="utf-8")
df["tasa_tx_mbps"] = 8 * df["bytes_tx"] / (df["duracion_s"] * 1e6)

print("\n--- Estructura del DataFrame (df.info()) ---")
print(df.info())

print("\n--- Conteo de Valores Nulos (df.isna().sum()) ---")
print(df.isna().sum())

print("\n--- Resumen Agrupado por Escenario ---")
resumen = df.groupby("escenario")[["tasa_tx_mbps", "rtt_medio_ms", "perdida_pct"]].agg(["median", "min", "max"])
print(resumen)

# -------------------------------------------------------------
# GRÁFICO 1: Dispersión Tasa TX vs RTT (Color = Pérdida)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(
    df["tasa_tx_mbps"],
    df["rtt_medio_ms"],
    c=df["perdida_pct"],
    cmap="viridis",
    s=80,
    edgecolors="black",
    linewidths=0.8,
    alpha=0.9
)
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Pérdida de paquetes (%)", fontsize=11, fontweight="bold")

ax.set_title("Relación entre Tasa TX y Latencia Media (RTT)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Tasa TX medida (Mbps)", fontsize=11, fontweight="bold")
ax.set_ylabel("RTT medio (ms)", fontsize=11, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)

# Anotación del codo o umbral de congestión
ax.annotate(
    "Umbral de Congestión\n(Quiebre de Linealidad)",
    xy=(80, 40),
    xytext=(50, 95),
    arrowprops=dict(facecolor="#d9534f", shrink=0.08, width=1.5, headwidth=8),
    fontsize=10,
    fontweight="bold",
    color="#d9534f",
    bbox=dict(boxstyle="round,pad=0.3", fc="#fdf7f7", ec="#d9534f", lw=1)
)

fig.tight_layout()
g1_path = os.path.join(GRAFICOS_DIR, "grafico1_dispersion_tasa_vs_rtt.png")
fig.savefig(g1_path, dpi=300)
plt.close(fig)
print(f"\n[Guardado] Gráfico 1 en: {g1_path}")

# -------------------------------------------------------------
# GRÁFICO 2: RTT y Pérdida a lo largo del Tiempo (Series temporales)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

df_sorted = df.sort_values(by="inicio").reset_index(drop=True)

ax1.plot(df_sorted["ventana_id"], df_sorted["rtt_medio_ms"], marker="o", color="#1f77b4", lw=2, label="RTT Medio (ms)")
ax1.fill_between(df_sorted["ventana_id"], df_sorted["rtt_min_ms"], df_sorted["rtt_max_ms"], color="#1f77b4", alpha=0.2, label="Rango [Min, Max]")
ax1.set_ylabel("RTT (ms)", fontsize=11, fontweight="bold")
ax1.set_title("Evolución Temporal de la Latencia y Pérdida por Ventana", fontsize=13, fontweight="bold", pad=10)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper left")

ax2.bar(df_sorted["ventana_id"], df_sorted["perdida_pct"], color="#d62728", alpha=0.7, width=0.6, label="Pérdida (%)")
ax2.set_xlabel("Número Secuencial de Ventana (60 s c/u)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Pérdida (%)", fontsize=11, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper left")

fig.tight_layout()
g2_path = os.path.join(GRAFICOS_DIR, "grafico2_evolucion_temporal.png")
fig.savefig(g2_path, dpi=300)
plt.close(fig)
print(f"[Guardado] Gráfico 2 en: {g2_path}")

# -------------------------------------------------------------
# GRÁFICO 3: Diagrama de Caja (Boxplot) de RTT por Escenario
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

escenarios_unicos = df["escenario"].unique()
datos_boxplot = [df[df["escenario"] == esc]["rtt_medio_ms"] for esc in escenarios_unicos]
etiquetas = [esc.replace("0", "").replace("_", " ").title() for esc in escenarios_unicos]

bp = ax.boxplot(
    datos_boxplot,
    tick_labels=etiquetas,
    patch_artist=True,
    notch=False,
    boxprops=dict(facecolor="#cce5ff", color="#1b4f72", lw=1.2),
    capprops=dict(color="#1b4f72", lw=1.2),
    whiskerprops=dict(color="#1b4f72", lw=1.2, linestyle="--"),
    flierprops=dict(marker="o", color="#d9534f", alpha=0.6),
    medianprops=dict(color="#b03a2e", lw=2)
)

ax.set_title("Diagrama de Caja de RTT Medio por Nivel de Carga", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Escenario de Tráfico", fontsize=11, fontweight="bold")
ax.set_ylabel("RTT Medio (ms)", fontsize=11, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
g3_path = os.path.join(GRAFICOS_DIR, "grafico3_boxplot_rtt.png")
fig.savefig(g3_path, dpi=300)
plt.close(fig)
print(f"[Guardado] Gráfico 3 en: {g3_path}")

# -------------------------------------------------------------
# GRÁFICO 4: Histograma de Latencia (Distribución y Asimetría)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))

ax.hist(df[df["escenario"] == "00_base_0M"]["rtt_medio_ms"], bins=6, alpha=0.6, color="#2ca02c", label="Línea Base (0M)", edgecolor="black")
ax.hist(df[df["escenario"].isin(["01_carga_20M", "02_carga_40M", "03_carga_60M"])]["rtt_medio_ms"], bins=8, alpha=0.6, color="#1f77b4", label="Carga Media (20-60M)", edgecolor="black")
ax.hist(df[df["escenario"].isin(["04_carga_80M", "05_congestion_100M"])]["rtt_medio_ms"], bins=8, alpha=0.7, color="#d62728", label="Saturación (80-100M)", edgecolor="black")

ax.set_title("Histograma de Distribución de RTT por Régimen de Red", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("RTT Medio (ms)", fontsize=11, fontweight="bold")
ax.set_ylabel("Frecuencia (Ventanas)", fontsize=11, fontweight="bold")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
g4_path = os.path.join(GRAFICOS_DIR, "grafico4_histograma_rtt.png")
fig.savefig(g4_path, dpi=300)
plt.close(fig)
print(f"[Guardado] Gráfico 4 en: {g4_path}")

# -------------------------------------------------------------
# CONCLUSIONES PARA EL INFORME
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("INTERPRETACIÓN DE RESULTADOS PARA EL INFORME (PUNTO 4)")
print("=" * 70)
print("""
1. ¿Aparece una cola en la distribución?
   Sí. En el histograma y en los rangos máximos se observa una asimetría hacia
   la derecha (cola larga). En reposo la latencia es compacta (~14 ms), pero al
   introducir carga aparecen ráfagas con picos de RTT que alcanzan hasta 216 ms
   y 950 ms.

2. ¿Hay dispersión creciente?
   Sí. En el diagrama de caja se aprecia claramente cómo la altura de las cajas
   y los bigotes se ensanchan a medida que aumenta la tasa transmitida. A cargas
   bajas (0-40 Mbps) la varianza es muy baja; pero por encima de 80 Mbps el RTT
   oscila violentamente debido al vaciado y llenado intermitente de las colas.

3. ¿Existe un umbral (knee-point) a partir del cual aumenta la latencia?
   Sí, existe un umbral crítico alrededor de los 75-80 Mbps. 
   - Entre 0 y 60 Mbps, el RTT crece suavemente de forma lineal (de 14 a 25 ms).
   - Superados los 80 Mbps, el enlace entra en congestión severa: los búferes
     del switch/AP colapsan (bufferbloat), la pérdida salta al 22% y el RTT
     se dispara de forma no lineal hasta los 122-142 ms.
""")
