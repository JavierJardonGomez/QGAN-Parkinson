import os
from pathlib import Path
import argparse
import numpy as np
import torch
import torchaudio

from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)

from torcheval.metrics import FrechetAudioDistance

def preprocess_audio(audio: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    """→ mono, resample, normaliza a ±1."""
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
    max_val = torch.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def stem_key(path: Path) -> str:
    """'paciente01_var2'-> 'paciente01'"""
    return path.stem.split("_")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--sr", type=int, choices=[16000, 22050, 24000], default=16000)
    parser.add_argument("--mode", choices=["nb", "wb"], default="nb")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    torch.manual_seed(0)

    real_files = [Path(args.real_dir) / f for f in os.listdir(args.real_dir) if f.endswith(".wav")]
    gen_files = [Path(args.gen_dir) / f for f in os.listdir(args.gen_dir) if f.endswith(".wav")]

    variants = {}
    for g in gen_files:
        variants.setdefault(stem_key(g), []).append(g)

    pesq_metric = PerceptualEvaluationSpeechQuality(args.sr, args.mode).to(device)
    stoi_metric = ShortTimeObjectiveIntelligibility(args.sr).to(device)
    fad_metric = FrechetAudioDistance.with_vggish(device=device)

    pesq_scores, stoi_scores = [], []
    skipped = []

    for ref_path in real_files:
        stem = stem_key(ref_path)
        if stem not in variants:
            skipped.append((ref_path, None, "sin variantes"))
            continue

        ref_audio, sr_ref = torchaudio.load(ref_path)
        ref_audio = preprocess_audio(ref_audio, sr_ref, args.sr).to(device)

        for gen_path in variants[stem]:
            try:
                deg_audio, sr_deg = torchaudio.load(gen_path)
                deg_audio = preprocess_audio(deg_audio, sr_deg, args.sr).to(device)

                # recorta o rellena para igualar longitud
                min_len = min(ref_audio.shape[-1], deg_audio.shape[-1])
                if min_len < args.sr // 4:
                    raise ValueError("audio <250 ms")

                ref_a = ref_audio[..., :min_len]
                deg_a = deg_audio[..., :min_len]

                pesq_res = pesq_metric(preds=deg_a, target=ref_a).item()
                stoi_res = stoi_metric(preds=deg_a, target=ref_a).item()

                pesq_scores.append(pesq_res)
                stoi_scores.append(stoi_res)

                # FAD
                fad_metric.update(preds=deg_a, targets=ref_a)

            except Exception as e:
                skipped.append((ref_path, gen_path, str(e)))

    if pesq_scores:
        avg_pesq = float(np.mean(pesq_scores))
        avg_mos = (avg_pesq - 1) * 4 / 3.5 + 1
        avg_stoi = float(np.mean(stoi_scores))

        try:
            fad_value = fad_metric.compute().item()
        except Exception as e:
            fad_value = None
            fad_error = str(e)

        print(f"\nEvaluadas {len(pesq_scores)} parejas")
        print(f"Average PESQ: {avg_pesq:.4f}")
        print(f"Approximated MOS: {avg_mos:.4f}")
        print(f"Average STOI: {avg_stoi:.4f}")
        if fad_value is not None:
            print(f"FAD: {fad_value:.4f}")
        else:
            print(f"FAD: --  (error: {fad_error})")
    else:
        print("No se calculo nada.")

    if skipped:
        print("\nOmitidos:")
        for r, g, e in skipped:
            g_name = Path(g).name if g else "---"
            print(f"  Ref: {Path(r).name} | Deg: {g_name} | {e}")


if __name__ == "__main__":
    main()