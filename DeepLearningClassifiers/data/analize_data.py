import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_audio_durations(audio_dir, plot_histogram=True):
    durations = []

    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(audio_dir, filename)
            try:
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                durations.append(duration)
            except Exception as e:
                print(f"Error con {filename}: {e}")

    if not durations:
        print("No se encontraron audios válidos.")
        return

    durations = np.array(durations)
    print(f"📁 Audios procesados: {len(durations)}")
    print(f"⏱️ Duración media: {durations.mean():.2f} s")
    print(f"⏳ Duración mínima: {durations.min():.2f} s")
    print(f"⏱️ Duración máxima: {durations.max():.2f} s")
    print(f"📊 Desviación estándar: {durations.std():.2f} s")

    if plot_histogram:
        plt.hist(durations, bins=20, edgecolor='black')
        plt.title("Distribución de duraciones de audio")
        plt.xlabel("Duración (segundos)")
        plt.ylabel("Cantidad de audios")
        plt.grid(True)
        plt.show()

import os
import librosa

def count_audios_by_duration(audio_dir, threshold_seconds=2.32):
    shorter = 0
    longer = 0
    total = 0

    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(audio_dir, filename)
            try:
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                total += 1
                if duration < threshold_seconds:
                    shorter += 1
                else:
                    longer += 1
            except Exception as e:
                print(f"Error con {filename}: {e}")

    if total == 0:
        print("No se encontraron audios.")
        return

    print(f"🔍 Umbral: {threshold_seconds:.2f} segundos")
    print(f"🎧 Total de audios: {total}")
    print(f"⏱️ Más cortos que el umbral: {shorter} ({(shorter/total)*100:.2f}%)")
    print(f"⏳ Más largos o iguales: {longer} ({(longer/total)*100:.2f}%)")


if __name__ == "__main__":
    # Cambia el directorio según sea necesario
    analyze_audio_durations('./audios_e_batch_4')
    count_audios_by_duration("./audios_e_batch_4", threshold_seconds=2.32)
