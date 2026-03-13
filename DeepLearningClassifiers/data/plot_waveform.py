import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Parámetros ---
audio_path = "./vowels_e/AVPEPUDEAC0057e1_c.wav"  # Cambia al audio que quieras
audio_path = "./wavs_e_22050Hz_normalized_057/AVPEPUDEA0024e1_p.wav"  # Cambia al audio que quieras
audio_path = "./V5_inference_057/AVPEPUDEA0024e1_p_aug1_generated.wav"  # Cambia al audio que quieras
duration_sec = 0.57

# --- Cargar y procesar ---
audio, sr = librosa.load(audio_path, sr=None)
max_samples = int(duration_sec * sr)
audio = audio[:max_samples]
audio = audio / np.max(np.abs(audio))  # Normalización

# --- Tiempo para eje x ---
time = np.linspace(0, len(audio)/sr, num=len(audio))

# --- Plot ---
plt.figure(figsize=(12, 4))
plt.plot(time, audio, color='steelblue', alpha=0.7, linewidth=1)
plt.title("Forma de onda sintética (patológica)", fontsize=16, fontweight='bold')
plt.xlabel("Tiempo (segundos)", fontsize=14)
plt.ylabel("Amplitud (Normalizada)", fontsize=14)
plt.ylim(-1.05, 1.05)  # Escala uniforme para comparar
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# --- Guardar ---
output_image_path = os.path.basename(os.path.splitext(audio_path)[0]) + "_waveform.png"
plt.savefig(output_image_path, dpi=300)
plt.close()

print(f"✅ Imagen guardada como: {output_image_path}")
