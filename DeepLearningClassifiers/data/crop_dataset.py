#!/usr/bin/env python3
# crop_dataset.py
"""
Recorta todos los .wav de una carpeta a una duración fija (0,57 s) y
los guarda en una carpeta de destino manteniendo el mismo nombre.

Ejemplos:
    python crop_dataset.py --input_dir ./input_full_e --output_dir ./input_full_e_front_0p37 --mode front
    python crop_dataset.py -i ./audios -o ./audios_center -m center
"""

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def crop_audio(in_path: Path, out_path: Path, *,
               duration: float = 0.57,
               mode: str = "front",
               pad_short: bool = True) -> None:
    """Carga un wav, lo recorta/pad a *duration* y lo escribe en out_path."""
    y, sr = librosa.load(in_path, sr=None)           # mantiene el SR original
    target_len = int(duration * sr)

    if len(y) < target_len:
        if not pad_short:
            # Si no quieres rellenar, simplemente omite el archivo corto
            return
        y = np.pad(y, (0, target_len - len(y)))
    else:
        if mode == "front":
            y = y[:target_len]
        elif mode == "center":
            start = (len(y) - target_len) // 2
            y = y[start:start + target_len]
        else:
            raise ValueError("mode debe ser 'front' o 'center'")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y, sr)


def main():
    parser = argparse.ArgumentParser(description="Recorta un dataset de audio a una duración fija")
    parser.add_argument("-i", "--input_dir", required=True, help="Carpeta con los wav originales")
    parser.add_argument("-o", "--output_dir", required=True, help="Carpeta destino para los wav recortados")
    parser.add_argument("-m", "--mode", choices=["front", "center"], default="front",
                        help="Dónde recortar: 'front' = desde el inicio, 'center' = centrado")
    parser.add_argument("--duration", type=float, default=0.57, help="Duración en segundos (defecto 0.37)")
    parser.add_argument("--no_pad", action="store_true", help="No rellenar audios más cortos que la duración objetivo")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    wav_files = sorted(in_dir.glob("*.wav"))

    if not wav_files:
        print(f"No se encontraron WAV en {in_dir}")
        return

    for wav in wav_files:
        out_path = out_dir / wav.name
        crop_audio(wav, out_path,
                   duration=args.duration,
                   mode=args.mode,
                   pad_short=not args.no_pad)

    print(f"Procesados {len(wav_files)} archivos. Salida en {out_dir}")


if __name__ == "__main__":
    main()
