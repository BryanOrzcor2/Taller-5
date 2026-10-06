# 📡 Taller 5: Análisis de Tráfico LAN y Predicción bajo Congestión

**Maestría en Inteligencia Artificial**  
**Universidad Sergio Arboleda**  
**Materia:** Introducción a la Inteligencia Artificial  
**Autor:** Bryan Orozco Romero  

---

## 📌 Descripción del Proyecto
Este repositorio contiene el desarrollo, scripts de procesamiento, conjuntos de datos y análisis exploratorio de datos (EDA) para el **Taller 5**, enfocado en el monitoreo y análisis de tráfico en redes LAN bajo diferentes escenarios de congestión y carga de red.

## 📂 Estructura del Repositorio
```text
├── Cargas A1/                 # Trazas de tráfico e iperf del Equipo A1
├── Cargas A2/                 # Trazas de tráfico e iperf del Equipo A2
├── Datos del Servidor/        # Métricas y registros del servidor central
├── Ping Equipo A1/            # Mediciones de latencia y RTT base y con carga (A1)
├── Ping Equipo A2/            # Mediciones de latencia y RTT base y con carga (A2)
├── graficos/                  # Visualizaciones y gráficos estadísticos generados
├── construir_dataset.py       # Pipeline de extracción, limpieza y construcción de ventanas.csv
├── exploracion_eda.py         # Análisis exploratorio, correlaciones y generación de figuras
├── ventanas.csv               # Dataset consolidado por ventanas de tiempo
└── Taller_5_MIA.pdf           # Guía y enunciado del Taller 5
```

## 🚀 Ejecución y Uso

1. **Construcción del Dataset:**
   ```bash
   python construir_dataset.py
   ```
   Procesa los archivos de telemetría de red, pings y logs de `iperf` para estructurar el archivo `ventanas.csv`.

2. **Análisis Exploratorio (EDA):**
   ```bash
   python exploracion_eda.py
   ```
   Genera las métricas descriptivas y las gráficas en el directorio `graficos/`.

---
*Universidad Sergio Arboleda — Escuela de Ciencias Exactas e Ingeniería*
