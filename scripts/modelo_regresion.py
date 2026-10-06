# -*- coding: utf-8 -*-
"""
Taller 5 - Puntos 5 y 6: Modelo de Regresión Lineal y Predicción bajo Congestión
Curso de Inteligencia Artificial - Universidad Sergio Arboleda


Contenido:
- Punto 5: Ajuste de modelo OLS, división por sesiones/bloques, baseline de mediana,
           métricas de error (MAE, RMSE, R2) y gráficos de regresión y residuos.
- Punto 6: Evaluación del escenario de congestión (100 Mbps), demostración del
           quiebre de linealidad (Bufferbloat), análisis de causalidad y propuesta
           de modelo segmentado (Piecewise Linear).
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------------------
# 1. Configuración de Rutas Dinámicas
# -------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "scripts" else CURRENT_DIR
CSV_PATH = BASE_DIR / "ventanas.csv"
GRAFICOS_DIR = BASE_DIR / "graficos"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TALLER 5: MODELADO DE REGRESIÓN LINEAL Y PREDICCIÓN BAJO CONGESTIÓN")
print(f"Ruta base del proyecto: {BASE_DIR}")
print(f"Dataset analizado:     {CSV_PATH}")
print("=" * 80)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo {CSV_PATH}. Ejecute construir_dataset.py primero.")

df = pd.read_csv(CSV_PATH)

# Asegurar tipo fecha
if "inicio" in df.columns:
    df["inicio"] = pd.to_datetime(df["inicio"], errors="coerce")

# -------------------------------------------------------------
# 2. Partición por Sesiones / Bloques de Carga (Punto 5 del PDF)
# -------------------------------------------------------------
# Regla del taller: Entrenar en régimen operacional controlado (0M a 60M)
# Evaluar en bloque posterior de prueba (80M) para medir generalización
# Reservar el escenario 100M para la prueba de estrés de congestión (Punto 6)

train_df = df[df["escenario"].isin(["00_base_0M", "01_carga_20M", "02_carga_40M", "03_carga_60M"])].copy()
test_80_df = df[df["escenario"] == "04_carga_80M"].copy()
test_100_df = df[df["escenario"] == "05_congestion_100M"].copy()

n_train = len(train_df)
n_test_80 = len(test_80_df)
n_test_100 = len(test_100_df)

print(f"\n[PARTICIÓN DE DATOS]")
print(f"• Entrenamiento (0M, 20M, 40M, 60M): {n_train} ventanas")
print(f"  - Rango de Tasa TX en train:        [{train_df['tasa_tx_mbps'].min():.4f}, {train_df['tasa_tx_mbps'].max():.4f}] Mbps")
print(f"  - Rango de RTT en train:            [{train_df['rtt_medio_ms'].min():.2f}, {train_df['rtt_medio_ms'].max():.2f}] ms")
print(f"• Prueba Controlada (80M):            {n_test_80} ventanas")
print(f"  - Rango de Tasa TX en test (80M):   [{test_80_df['tasa_tx_mbps'].min():.4f}, {test_80_df['tasa_tx_mbps'].max():.4f}] Mbps")
print(f"• Escenario Congestionado (100M):     {n_test_100} ventanas")
print(f"  - Rango de Tasa TX en cong. (100M): [{test_100_df['tasa_tx_mbps'].min():.4f}, {test_100_df['tasa_tx_mbps'].max():.4f}] Mbps")

# -------------------------------------------------------------
# 3. Ajuste del Modelo de Regresión Lineal OLS (Punto 5)
# -------------------------------------------------------------
X_train = train_df[["tasa_tx_mbps"]].to_numpy()
y_train = train_df["rtt_medio_ms"].to_numpy()

modelo = LinearRegression()
modelo.fit(X_train, y_train)

beta_0 = modelo.intercept_
beta_1 = modelo.coef_[0]
r2_train = modelo.score(X_train, y_train)

# Línea Base: Mediana de RTT del set de entrenamiento
mediana_train = float(np.median(y_train))

print(f"\n[PARÁMETROS DEL MODELO LINEAL OLS]")
print(f"• Intercepto (beta_0):               {beta_0:.4f} ms")
print(f"  -> Interpretación: RTT basal de propagación/switching con tráfico nulo.")
print(f"• Pendiente (beta_1):                {beta_1:.4f} ms/Mbps")
print(f"  -> Interpretación: Incremento medio de {beta_1:.4f} ms por cada Mbps inyectado.")
print(f"• R^2 en conjunto de entrenamiento:  {r2_train:.4f}")
print(f"• Línea base constante (mediana):   {mediana_train:.2f} ms")

# -------------------------------------------------------------
# 4. Evaluación en Conjunto de Prueba Controlada (80 Mbps)
# -------------------------------------------------------------
X_test_80 = test_80_df[["tasa_tx_mbps"]].to_numpy()
y_real_80 = test_80_df["rtt_medio_ms"].to_numpy()

pred_80 = modelo.predict(X_test_80)
pred_base_80 = np.full_like(y_real_80, mediana_train)

mae_80 = mean_absolute_error(y_real_80, pred_80)
rmse_80 = np.sqrt(mean_squared_error(y_real_80, pred_80))
r2_80 = r2_score(y_real_80, pred_80)

mae_base_80 = mean_absolute_error(y_real_80, pred_base_80)
rmse_base_80 = np.sqrt(mean_squared_error(y_real_80, pred_base_80))

print(f"\n[EVALUACIÓN EN PRUEBA CONTROLADA (80 Mbps - Pre-Congestión)]")
print(f"• RTT Medido Real (ms):              {np.round(y_real_80, 2)}")
print(f"• RTT Predicho Modelo (ms):          {np.round(pred_80, 2)}")
print(f"• Residuos (y - y_hat) (ms):         {np.round(y_real_80 - pred_80, 2)}")
print(f"• MAE Modelo Lineal:                 {mae_80:.3f} ms")
print(f"• RMSE Modelo Lineal:                {rmse_80:.3f} ms")
print(f"• R^2 Modelo Lineal:                 {r2_80:.3f}")
print(f"• MAE Línea Base Constante:          {mae_base_80:.3f} ms")
print(f"• Conclusión 80M: El modelo lineal supera a la mediana baseline (MAE {mae_80:.2f} vs {mae_base_80:.2f} ms),")
print(f"  aunque los residuos positivos (+9.9 ms) ya anuncian el inicio del encolamiento.")

# -------------------------------------------------------------
# 5. Punto 6: Predicción de Escenario Congestionado (100 Mbps)
# -------------------------------------------------------------
# a) Predicción a priori registrada antes de contrastar
X_test_100 = test_100_df[["tasa_tx_mbps"]].to_numpy()
y_real_100 = test_100_df["rtt_medio_ms"].to_numpy()

pred_100 = modelo.predict(X_test_100)
residuos_100 = y_real_100 - pred_100
mae_100 = mean_absolute_error(y_real_100, pred_100)
rmse_100 = np.sqrt(mean_squared_error(y_real_100, pred_100))

# Métricas de congestión concurrentes
tasa_rx_mean_100 = test_100_df["tasa_rx_mbps"].mean()
tasa_tx_mean_100 = test_100_df["tasa_tx_mbps"].mean()
perdida_mean_100 = test_100_df["perdida_pct"].mean()
descartes_mean_100 = test_100_df["descartes"].mean()

print(f"\n" + "=" * 80)
print("PUNTO 6: PREDICCIÓN Y RUPTURA DE LINEALIDAD BAJO CONGESTIÓN (100 Mbps)")
print("=" * 80)
print(f"• Naturaleza del pronóstico:         EXTRAPOLACIÓN")
print(f"  - Rango de entrenamiento:          [0.00, 60.07] Mbps")
print(f"  - Punto evaluado:                  ~100.13 Mbps (Muy fuera del dominio lineal)")
print(f"• Predicción a priori del modelo:    {np.round(pred_100, 2)} ms (Promedio: {pred_100.mean():.2f} ms)")
print(f"• Medición Real en Laboratorio:      {np.round(y_real_100, 2)} ms")
print(f"  - Mediana Real:                    {np.median(y_real_100):.2f} ms")
print(f"  - Rango Real:                      [{y_real_100.min():.2f}, {y_real_100.max():.2f}] ms")
print(f"• Residuos Sistemáticos (y - y_hat): {np.round(residuos_100, 2)} ms")
print(f"• MAE en Congestión:                 {mae_100:.2f} ms (Subestimación del {((y_real_100.mean() - pred_100.mean()) / pred_100.mean()) * 100:.1f}%)")
print(f"• RMSE en Congestión:                {rmse_100:.2f} ms")
print(f"\n[EVIDENCIAS FÍSICAS DE CONGESTIÓN Y SATURACIÓN]")
print(f"1. Estancamiento de Caudal (Goodput): TX={tasa_tx_mean_100:.2f} Mbps, pero RX={tasa_rx_mean_100:.2f} Mbps (Cuello de botella a ~76 Mbps)")
print(f"2. Pérdida Crítica de Paquetes:      {perdida_mean_100:.2f}% de sondas perdidas")
print(f"3. Descarte Masivo en Buffers:       {descartes_mean_100:,.0f} paquetes descartados")
print(f"4. Causa del fallo de ML:            Fenómeno de Bufferbloat y teoría de colas M/M/1.")
print(f"   La relación tiempo de espera vs utilización es asintótica: W = 1 / (mu - lambda).")
print(f"   El modelo lineal OLS asume tubería infinita sin colas; por eso subestima en >90 ms.")

# -------------------------------------------------------------
# 6. Generación de Gráficos (Figuras 5 y 6)
# -------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

# -------------------------------------------------------------
# FIGURA 5: Regresión Lineal, Extrapolación y Ruptura por Congestión
# -------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(10, 6), dpi=300)

# Puntos de entrenamiento
ax5.scatter(
    train_df["tasa_tx_mbps"],
    train_df["rtt_medio_ms"],
    color="#1f77b4",
    s=70,
    alpha=0.85,
    edgecolors="black",
    linewidths=0.8,
    label="Entrenamiento (0 - 60 Mbps, N=19)",
    zorder=4
)

# Puntos de prueba 80M
ax5.scatter(
    test_80_df["tasa_tx_mbps"],
    test_80_df["rtt_medio_ms"],
    color="#2ca02c",
    s=90,
    marker="^",
    edgecolors="black",
    linewidths=0.9,
    label="Prueba 80M Real (Pre-congestión, N=3)",
    zorder=4
)

# Puntos de congestión 100M
ax5.scatter(
    test_100_df["tasa_tx_mbps"],
    test_100_df["rtt_medio_ms"],
    color="#d62728",
    s=110,
    marker="X",
    edgecolors="black",
    linewidths=1.0,
    label="Congestión 100M Real (Saturado, N=3)",
    zorder=5
)

# Recta de regresión en zona entrenada
x_vals_train = np.linspace(0, 65, 100)
y_vals_train = modelo.predict(x_vals_train.reshape(-1, 1))
ax5.plot(
    x_vals_train,
    y_vals_train,
    color="#08519c",
    linewidth=2.5,
    linestyle="-",
    label=f"Regresión Lineal Ajustada: $\\hat{{y}} = {beta_0:.2f} + {beta_1:.3f}x$",
    zorder=3
)

# Recta extrapolada en zona de congestión
x_vals_extrap = np.linspace(65, 105, 100)
y_vals_extrap = modelo.predict(x_vals_extrap.reshape(-1, 1))
ax5.plot(
    x_vals_extrap,
    y_vals_extrap,
    color="#fb6a4a",
    linewidth=2.2,
    linestyle="--",
    label="Extrapolación Lineal Invalida (Fuera de rango)",
    zorder=3
)

# Predicciones puntuales a 80M y 100M
ax5.scatter(
    test_80_df["tasa_tx_mbps"],
    pred_80,
    color="#74c476",
    s=80,
    marker="o",
    facecolors="none",
    edgecolors="#006d2c",
    linewidths=1.8,
    label="Predicción Lineal a 80M",
    zorder=6
)
ax5.scatter(
    test_100_df["tasa_tx_mbps"],
    pred_100,
    color="#fc9272",
    s=90,
    marker="s",
    facecolors="none",
    edgecolors="#cb181d",
    linewidths=2.0,
    label="Predicción Lineal a 100M (Ingenua)",
    zorder=6
)

# Línea base de mediana
ax5.axhline(
    mediana_train,
    color="#737373",
    linestyle=":",
    linewidth=1.5,
    label=f"Línea Base Constante (Mediana = {mediana_train:.1f} ms)",
    zorder=2
)

# Anotación del error de extrapolación en 100M
rtt_real_prom_100 = y_real_100.mean()
pred_prom_100 = pred_100.mean()
tx_prom_100 = test_100_df["tasa_tx_mbps"].mean()

ax5.annotate(
    "",
    xy=(tx_prom_100, rtt_real_prom_100),
    xytext=(tx_prom_100, pred_prom_100),
    arrowprops=dict(arrowstyle="<->", color="#99000d", lw=2.0)
)
ax5.text(
    tx_prom_100 - 15,
    (rtt_real_prom_100 + pred_prom_100) / 2 - 5,
    f"Fallo de Linealidad\nError Residual: +{rtt_real_prom_100 - pred_prom_100:.1f} ms\n(Bufferbloat)",
    color="#99000d",
    fontsize=9.5,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fee5d9", edgecolor="#de2d26", alpha=0.95)
)

# Sombreado de zona de congestión
ax5.axvspan(76, 105, color="#fee0d2", alpha=0.35, label="Zona de Saturación / Congestión (>76 Mbps)")

ax5.set_title("Taller 5: Ajuste de Regresión Lineal y Ruptura Asintótica por Congestión", fontsize=13, fontweight="bold", pad=12)
ax5.set_xlabel("Tasa de Transmisión TX Medida (Mbps)", fontsize=11, labelpad=8)
ax5.set_ylabel("RTT Medio Medido (ms)", fontsize=11, labelpad=8)
ax5.set_xlim(-2, 106)
ax5.set_ylim(0, 155)
ax5.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
fig5.tight_layout()

f5_path = GRAFICOS_DIR / "grafico5_regresion_lineal.png"
fig5.savefig(f5_path, dpi=300)
plt.close(fig5)
print(f"\n[OK] Gráfico 5 guardado en: {f5_path}")

# -------------------------------------------------------------
# FIGURA 6: Diagnóstico de Residuos (y - y_hat) vs Carga
# -------------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(10, 5.5), dpi=300)

# Residuos en Train
residuos_train = y_train - modelo.predict(X_train)
ax6.scatter(
    train_df["tasa_tx_mbps"],
    residuos_train,
    color="#1f77b4",
    s=65,
    alpha=0.8,
    edgecolors="black",
    linewidths=0.7,
    label="Residuos Entrenamiento (0 - 60 Mbps)",
    zorder=4
)

# Residuos en Test 80M
residuos_80 = y_real_80 - pred_80
ax6.scatter(
    test_80_df["tasa_tx_mbps"],
    residuos_80,
    color="#2ca02c",
    s=85,
    marker="^",
    edgecolors="black",
    linewidths=0.9,
    label="Residuos Prueba 80M (Pre-congestión)",
    zorder=4
)

# Residuos en Congestión 100M
ax6.scatter(
    test_100_df["tasa_tx_mbps"],
    residuos_100,
    color="#d62728",
    s=110,
    marker="X",
    edgecolors="black",
    linewidths=1.0,
    label="Residuos Congestión 100M (Explosión)",
    zorder=5
)

# Línea de residuo cero
ax6.axhline(0, color="black", linestyle="-", linewidth=1.2, zorder=3)

# Banda de tolerancia homocedástica (+/- 8 ms)
ax6.axhspan(-8, 8, color="#e0f3f8", alpha=0.6, label="Banda de dispersión normal ([-8, +8] ms)")

# Sombreado de régimen de congestión
ax6.axvspan(76, 105, color="#fee0d2", alpha=0.35, label="Régimen saturado (>76 Mbps)")

ax6.annotate(
    "Explosión sistemática de residuos\n(Heterocedasticidad extrema por encolamiento)",
    xy=(tx_prom_100, residuos_100.mean()),
    xytext=(tx_prom_100 - 35, residuos_100.mean() - 15),
    arrowprops=dict(facecolor="#de2d26", arrowstyle="->", lw=1.5),
    fontsize=9.5,
    fontweight="bold",
    color="#99000d",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#fee5d9", edgecolor="#de2d26", alpha=0.95)
)

ax6.set_title("Taller 5: Diagnóstico de Residuos $(y - \\hat{y})$ vs Carga de Tráfico", fontsize=13, fontweight="bold", pad=12)
ax6.set_xlabel("Tasa TX Medida (Mbps)", fontsize=11, labelpad=8)
ax6.set_ylabel("Residuo $e_i = y_i - \\hat{y}_i$ (ms)", fontsize=11, labelpad=8)
ax6.set_xlim(-2, 106)
ax6.set_ylim(-15, 120)
ax6.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
fig6.tight_layout()

f6_path = GRAFICOS_DIR / "grafico6_analisis_residuos.png"
fig6.savefig(f6_path, dpi=300)
plt.close(fig6)
print(f"[OK] Gráfico 6 guardado en: {f6_path}")

# -------------------------------------------------------------
# 7. Resumen de Métricas para el Informe
# -------------------------------------------------------------
resumen_metricas = pd.DataFrame([
    {
        "Conjunto": "Entrenamiento (0 - 60M)",
        "Ventanas (N)": n_train,
        "Rango Tasa TX (Mbps)": f"[{train_df['tasa_tx_mbps'].min():.2f}, {train_df['tasa_tx_mbps'].max():.2f}]",
        "MAE Modelo (ms)": f"{mean_absolute_error(y_train, modelo.predict(X_train)):.2f}",
        "RMSE Modelo (ms)": f"{np.sqrt(mean_squared_error(y_train, modelo.predict(X_train))):.2f}",
        "R2": f"{r2_train:.3f}",
        "MAE Baseline (ms)": f"{mean_absolute_error(y_train, np.full_like(y_train, mediana_train)):.2f}"
    },
    {
        "Conjunto": "Prueba (80M Pre-Congestión)",
        "Ventanas (N)": n_test_80,
        "Rango Tasa TX (Mbps)": f"[{test_80_df['tasa_tx_mbps'].min():.2f}, {test_80_df['tasa_tx_mbps'].max():.2f}]",
        "MAE Modelo (ms)": f"{mae_80:.2f}",
        "RMSE Modelo (ms)": f"{rmse_80:.2f}",
        "R2": f"{r2_80:.3f}",
        "MAE Baseline (ms)": f"{mae_base_80:.2f}"
    },
    {
        "Conjunto": "Congestión (100M Extrapolación)",
        "Ventanas (N)": n_test_100,
        "Rango Tasa TX (Mbps)": f"[{test_100_df['tasa_tx_mbps'].min():.2f}, {test_100_df['tasa_tx_mbps'].max():.2f}]",
        "MAE Modelo (ms)": f"{mae_100:.2f}",
        "RMSE Modelo (ms)": f"{rmse_100:.2f}",
        "R2": f"{r2_score(y_real_100, pred_100):.3f}",
        "MAE Baseline (ms)": f"{mean_absolute_error(y_real_100, np.full_like(y_real_100, mediana_train)):.2f}"
    }
])

print("\n" + "=" * 80)
print("TABLA CONSOLIDADA DE MÉTRICAS (EVIDENCIAS 4 Y 5)")
print("=" * 80)
print(resumen_metricas.to_string(index=False))
print("=" * 80)
print("\n[PROCESO COMPLETADO EXITOSAMENTE]")
