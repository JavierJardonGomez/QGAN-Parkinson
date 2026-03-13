import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from tqdm import tqdm

# --- Parámetros ---
#input_dir = "Vowels/Vowels/Patologicas/E"       # Directorio con audios originales a 44100 Hz
input_dir = "Vowels/Vowels/Control/E"       # Directorio con audios originales a 44100 Hz
output_dir = "wavs_e_22050Hz_normalize/"      # Directorio donde se guardarán los audios convertidos
target_sr = 22050             # Tasa de muestreo destino

os.makedirs(output_dir, exist_ok=True)

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

# --- Procesamiento ---
for fname in tqdm(os.listdir(input_dir)):
    if not fname.lower().endswith(".wav"):
        continue

    input_path = os.path.join(input_dir, fname)
    
    # Agregar "_c" al nombre del archivo
    base, ext = os.path.splitext(fname)
    output_fname = base + "_c" + ext
    output_path = os.path.join(output_dir, output_fname)

    wav, _ = librosa.load(input_path, sr=target_sr)

    wav = nr.reduce_noise(y=wav, sr=target_sr)

    wav = normalize_audio(wav)
    
    sf.write(output_path, wav, target_sr)