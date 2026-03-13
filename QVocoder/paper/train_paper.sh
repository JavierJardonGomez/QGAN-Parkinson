#!/bin/bash
#Go to root
cd 
# Cambiar al directorio QVocoder
cd QVocoder || { echo "Error: No se pudo cambiar al directorio QVocoder"; exit 1; }

# Activar el entorno virtual
source .venv/bin/activate || { echo "Error: No se pudo activar el entorno virtual"; exit 1; }

# Cambiar al directorio qhifigan
cd qhifiGAN || { echo "Error: No se pudo cambiar al directorio qhifigan"; exit 1; }

# Ejecutar el script de entrenamiento
python train1Q_paper.py

