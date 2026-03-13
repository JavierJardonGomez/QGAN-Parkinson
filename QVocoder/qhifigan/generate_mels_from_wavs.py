import os
import torch
import numpy as np
from scipy.io.wavfile import read
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
from tqdm import tqdm

# --- Parámetros del modelo ---
n_fft = 1024
hop_size = 256
win_size = 1024
num_mels = 80
sampling_rate = 22050
fmin = 0
fmax = 8000

# --- Mel caching ---
mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

def mel_spectrogram(y, sampling_rate):
    global mel_basis, hann_window
    device = y.device
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + '_' + str(device)] = torch.from_numpy(mel).float().to(device)
        hann_window[str(device)] = torch.hann_window(win_size).to(device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(input=y, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
                      window=hann_window[str(device)], center=False,
                      pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(torch.pow(spec.real, 2) + torch.pow(spec.imag, 2) + 1e-9)

    mel_spec = torch.matmul(mel_basis[str(fmax) + '_' + str(device)], spec)
    mel_spec = spectral_normalize_torch(mel_spec)
    return mel_spec

# --- Ruta de entrada/salida ---
input_dir = "../wavs_e_22050Hz_normalize"
output_dir = "../mels_e_22050Hz_normalize"
os.makedirs(output_dir, exist_ok=True)

for fname in tqdm(os.listdir(input_dir)):
    if not fname.endswith(".wav"):
        continue

    wav_path = os.path.join(input_dir, fname)
    sampling_rate_file, wav = read(wav_path)
    if sampling_rate_file != sampling_rate:
        raise ValueError(f"Sample rate mismatch: expected {sampling_rate}, got {sampling_rate_file} in {fname}")

    wav = wav / 32768.0
    wav = normalize(wav) * 0.95
    wav = torch.FloatTensor(wav).unsqueeze(0).to("cpu")

    mel = mel_spectrogram(wav, sampling_rate).cpu().numpy()
    out_path = os.path.join(output_dir, fname.replace(".wav", ".npy"))
    np.save(out_path, mel)
