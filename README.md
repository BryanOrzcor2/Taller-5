# 📡 Taller 5: Análisis de Tráfico LAN y Predicción bajo Congestión

**Maestría en Inteligencia Artificial**  
**Universidad Sergio Arboleda**  
**Materia:** Introducción a la Inteligencia Artificial / Introducción a Machine Learning  
**Autor:** Bryan Orozco Romero  
**Docente:** Joaquín F. Sánchez  
**Repositorio GitHub:** [https://github.com/BryanOrzcor2/Taller-5](https://github.com/BryanOrzcor2/Taller-5)

---

## 📌 Descripción del Proyecto
Este proyecto contiene el ciclo completo de captura de telemetría de red, procesamiento de series temporales por ventanas de 60 segundos, análisis exploratorio de datos (EDA), y ajuste/evaluación de modelos de **Machine Learning Supervisado (Regresión Lineal OLS)** para anticipar la latencia de red (RTT) e identificar empíricamente la ruptura del supuesto de linealidad ante condiciones de congestión severa (*Bufferbloat*).

> **Nota de Topología Experimental Real:** El banco de pruebas se configuró con **tres equipos de cómputo físicos (dos clientes generadores A1 y A2, y un único servidor central receptor B)** interconectados en el mismo segmento de red local (`172.25.3.0/24`) **sin utilizar ningún switch de conmutación intermedio**. El cuello de botella físico y la saturación de búferes residen directamente en la tarjeta de red (NIC) y las colas de recepción del servidor B.

El desarrollo responde estrictamente a la guía y rúbrica de evaluación oficial de **5.0 puntos** contenida en `Taller_5_MIA.pdf`.

---

## 📂 Estructura del Repositorio
```text
├── Cargas A1/                 # Trazas de tráfico e iperf del Equipo A1
├── Cargas A2/                 # Trazas de tráfico e iperf del Equipo A2
├── Datos del Servidor/        # Métricas y registros del servidor central
├── Ping Equipo A1/            # Mediciones de latencia y RTT base y con carga (A1)
├── Ping Equipo A2/            # Mediciones de latencia y RTT base y con carga (A2)
├── graficos/                  # 6 visualizaciones estadísticas generadas (alta resolución)
│   ├── grafico1_dispersion_tasa_vs_rtt.png
│   ├── grafico2_evolucion_temporal.png
│   ├── grafico3_boxplot_rtt.png
│   ├── grafico4_histograma_rtt.png
│   ├── grafico5_regresion_lineal.png
│   └── grafico6_analisis_residuos.png
├── informe/                   # Informe final escrito (LaTeX / Overleaf)
│   ├── graficos/              # Copia local de figuras para compilación directa
│   └── informe_taller5.tex    # Documento de 5 páginas con la solución completa
├── scripts/                   # Códigos fuente reproducibles en Python
│   ├── construir_dataset.py   # Pipeline ETL: parseo de pings, iperf y contadores
│   ├── exploracion_eda.py     # Análisis exploratorio (EDA) y gráficos 1 al 4
│   └── modelo_regresion.py    # Modelo de regresión OLS, baseline y gráficos 5 y 6
├── ventanas.csv               # Dataset consolidado de 25 ventanas de medición
├── requirements.txt           # Dependencias de Python verificadas
├── README.md                  # Guía de ejecución y documentación técnica
└── Taller_5_MIA.pdf           # Guía oficial del taller
```

---

## ⚙️ Requisitos del Entorno

Instale las dependencias necesarias mediante `pip`:
```bash
pip install -r requirements.txt
```

Versiones principales utilizadas y verificadas:
* `python >= 3.10` (testeado en Python 3.14.3)
* `pandas >= 3.0.0`
* `numpy >= 2.0.0`
* `scikit-learn >= 1.9.0`
* `scipy >= 1.18.0`
* `matplotlib >= 3.11.0`

---

## 🚀 Guía de Ejecución Paso a Paso

Los scripts cuentan con resolución dinámica de rutas y pueden ejecutarse directamente desde la raíz del repositorio o dentro de la carpeta `scripts/`.

### 1. Construcción y Auditoría del Dataset
```bash
python scripts/construir_dataset.py
```
* **Entradas:** Archivos en `Ping Equipo A1/`, `Ping Equipo A2/`, `Cargas A1/`, `Cargas A2/`.
* **Salida:** `ventanas.csv` (25 ventanas estructuradas sin valores nulos).

### 2. Análisis Exploratorio de Datos (EDA)
```bash
python scripts/exploracion_eda.py
```
* **Salidas:** Genera los gráficos 1 al 4 en la carpeta `graficos/` y emite en consola el diagnóstico de colas largas, dispersión creciente y punto de inflexión (*knee-point*).

### 3. Ajuste de Regresión Lineal y Evaluación de Congestión
```bash
python scripts/modelo_regresion.py
```
* **Salidas:** Genera los gráficos 5 y 6 en `graficos/`, calcula los parámetros del modelo lineal $\hat{y} = \beta_0 + \beta_1 x$, evalúa el conjunto de prueba a 80 Mbps, contrasta contra la línea base de la mediana, y evalúa el fallo de extrapolación a 100 Mbps.

---

## 📊 Síntesis de Resultados del Modelo de Machine Learning

### Parámetros del Modelo Lineal OLS:
* **Intercepto ($\beta_0$):** $13.94\text{ ms}$ (RTT basal en reposo).
* **Pendiente ($\beta_1$):** $0.1755\text{ ms/Mbps}$ (costo marginal de latencia por cada Mbps).
* **Bondad de Ajuste en Entrenamiento ($R^2_{\text{train}}$):** $0.6359$.
* **Línea Base Constante (Mediana Train):** $16.00\text{ ms}$.

### Tabla Comparativa de Desempeño:
| Régimen / Partición | Ventanas ($N$) | Rango Tasa TX | MAE Modelo | RMSE Modelo | MAE Baseline | Diagnóstico Físico |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Entrenamiento ($0-60$M)** | 19 | $[0.00, 60.07]\text{ Mbps}$ | **$2.11\text{ ms}$** | $3.04\text{ ms}$ | $4.00\text{ ms}$ | Régimen lineal estable. |
| **Prueba ($80$M Pre-Cong.)** | 3 | $[79.97, 80.16]\text{ Mbps}$ | **$9.92\text{ ms}$** | $10.27\text{ ms}$ | $21.90\text{ ms}$ | Supera al baseline; inicio de encolamiento. |
| **Congestión ($100$M Extrap.)** | 3 | $[99.73, 100.67]\text{ Mbps}$ | **$90.49\text{ ms}$** | $92.26\text{ ms}$ | $106.00\text{ ms}$ | **Quiebre por Bufferbloat:** RTT real explota a $125.6\text{ ms}$ vs $31.5\text{ ms}$ predicho. Caudal estancado en $76.3\text{ Mbps}$, $22.2\%$ de pérdida y $113{,}577$ descartes. |

---

## 📄 Informe Final en LaTeX (5 Páginas)
El informe escrito se encuentra en:
`informe/informe_taller5.tex`

Está formateado en dos columnas con estilo formal académico e incluye:
1. Descripción reproducible del protocolo y topología de red.
2. Diccionario de variables y controles de calidad aplicados a `ventanas.csv`.
3. Análisis exploratorio e interpretación de las 6 figuras.
4. Evaluación del modelo de Machine Learning vs línea base y residuos.
5. Diagnóstico de congestión y propuesta de modelo segmentado (*Piecewise Linear*).
6. Respuestas rigurosas a las 4 preguntas de reflexión del docente.

Para compilar en **Overleaf**:
Basta con subir el contenido de la carpeta `informe/` (el archivo `.tex` y la carpeta `graficos/`). Compila automáticamente en pdfLaTeX sin dependencias externas complejas.

---
*Universidad Sergio Arboleda — Escuela de Ciencias Exactas e Ingeniería*
