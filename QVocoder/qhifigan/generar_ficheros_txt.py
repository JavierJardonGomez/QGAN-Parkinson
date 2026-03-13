import os

# Ruta al directorio con los archivos
directorio = "../pre_processed_input_wavs_e"

# Archivo de salida
archivo_salida = "training_e_c.txt"

# Listas separadas
archivos_c = []
archivos_p = []

# Clasificar archivos
for archivo in os.listdir(directorio):
    if archivo.endswith(".wav"):
        if archivo.endswith("_c.wav"):
            archivos_c.append(os.path.splitext(archivo)[0])
        elif archivo.endswith("_p.wav"):
            archivos_p.append(os.path.splitext(archivo)[0])

# Escribir en archivo
with open(archivo_salida, "w") as f:
    for nombre in sorted(archivos_c):
        f.write(f"{nombre}|\n")
    for nombre in sorted(archivos_p):
        f.write(f"{nombre}|\n")

print(f"Listado guardado en: {archivo_salida}")
