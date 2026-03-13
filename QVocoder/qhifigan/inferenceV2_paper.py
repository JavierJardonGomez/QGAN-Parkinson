
from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import os
import argparse
import json
import random

import torch
from scipy.io.wavfile import write
import torchaudio

from env import AttrDict
from meldatasetQ import mel_spectrogram, MAX_WAV_VALUE, load_wav
from modelsQ5B import QGenerator

h = None
device = None

# -------------------------------------------------------------------------
# utilidades de checkpoint
# -------------------------------------------------------------------------

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}' …")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

# -------------------------------------------------------------------------
#  Funciones de pre‑procesado
# -------------------------------------------------------------------------

def get_mel(x):
    return mel_spectrogram(
        x,
        h.n_fft,
        h.num_mels,
        h.sampling_rate,
        h.hop_size,
        h.win_size,
        h.fmin,
        h.fmax,
    )


def augment_mel(
    mel: torch.Tensor,
    *,
    noise_std: float = 5e-5,
    max_time_shift: int = 1,
    freq_mask_param: int = 1,
    time_mask_param: int = 1,
) -> torch.Tensor:
    """Perturbaciones suaves (modo *augment*)."""

    if noise_std > 0:
        mel = mel + torch.randn_like(mel) * noise_std

    if max_time_shift > 0:
        shift = random.randint(-max_time_shift, max_time_shift)
        if shift != 0:
            mel = torch.roll(mel, shifts=shift, dims=-1)

    if freq_mask_param > 0:
        mel = torchaudio.transforms.FrequencyMasking(freq_mask_param)(mel)
    if time_mask_param > 0:
        mel = torchaudio.transforms.TimeMasking(time_mask_param)(mel)

    return mel

# -------------------------------------------------------------------------
#  Inferencia
# -------------------------------------------------------------------------

def inference(args):
    generator = QGenerator(h).to(device)
    state_dict_g = load_checkpoint(args.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    # --- Selección modo de variación ---
    if args.variant_type == 'dropout':
        generator.train()  # Activa dropout / batch‑norm en modo train.
        for p in generator.parameters():
            p.requires_grad = False  # congelar pesos.
    else:
        generator.eval()

    filelist = os.listdir(args.input_wavs_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for fname in filelist:
            wav, _ = load_wav(os.path.join(args.input_wavs_dir, fname))
            wav = torch.FloatTensor(wav / MAX_WAV_VALUE).to(device)
            base_mel = get_mel(wav.unsqueeze(0))

            for v in range(args.n_variations):
                if args.variant_type == 'augment':
                    mel_variant = augment_mel(
                        base_mel,
                        noise_std=args.noise_std,
                        max_time_shift=args.max_time_shift,
                        freq_mask_param=args.freq_mask_param,
                        time_mask_param=args.time_mask_param,
                    )
                else:  # dropout || none
                    mel_variant = base_mel

                d1 = torchaudio.functional.compute_deltas(mel_variant)
                d2 = torchaudio.functional.compute_deltas(d1)
                d3 = torchaudio.functional.compute_deltas(d2)
                x = torch.cat([mel_variant, d1, d2, d3], dim=1)

                y_hat = generator(x)
                audio = (y_hat.squeeze() * MAX_WAV_VALUE).cpu().numpy().astype('int16')

                stem = os.path.splitext(fname)[0]
                tag = 'aug' if args.variant_type == 'augment' else 'do'
                out_path = os.path.join(args.output_dir, f"{stem}_{tag}{v+1}_generated.wav")
                write(out_path, h.sampling_rate, audio)
                print(out_path)

# -------------------------------------------------------------------------
#  CLI
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='../wavs_e_22050Hz_normalize')
    parser.add_argument('--output_dir', default='../experiment_cp_final_v2_mels_batch16_lr_g40_V5_inference')
    parser.add_argument('--checkpoint_file', required=True)

    parser.add_argument('--n_variations', type=int, default=3,
                        help='Copias sintéticas por wav')
    parser.add_argument('--variant_type', choices=['augment', 'dropout', 'none'],
                        default='augment', help='Estrategia de generación de variedad')

    # Hiperparámetros solo para variant_type=augment
    parser.add_argument('--noise_std', type=float, default=5e-5)
    parser.add_argument('--max_time_shift', type=int, default=1)
    parser.add_argument('--freq_mask_param', type=int, default=1)
    parser.add_argument('--time_mask_param', type=int, default=1)

    args = parser.parse_args()

    config_path = os.path.join(os.path.split(args.checkpoint_file)[0], 'config.json')
    with open(config_path) as f:
        cfg = json.load(f)

    global h, device
    h = AttrDict(cfg)
    torch.manual_seed(h.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
