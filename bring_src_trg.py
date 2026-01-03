import os 
import soundfile as sf 
from IPython.display import Audio, display
import json 
import numpy as np
import librosa
import torch

def rms_torch(x: torch.Tensor, eps=1e-9):
    """
    RMS 계산
    Args:
        x: Tensor (B, T) 또는 (B, 1, T)
    Returns:
        Tensor (B, 1)
    """
    if x.dim() == 3:  # (B, 1, T)
        x = x.squeeze(1)
    return torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)


def normalize(x: torch.Tensor, target_rms=0.1, eps=1e-9):
    """
    RMS 기반 waveform 정규화
    Args:
        x: Tensor (B, 1, T)
        target_rms: 목표 RMS (e.g., 0.1 ≈ -20dBFS)
    Returns:
        Tensor (B, 1, T)
    """
    rms_val = rms_torch(x, eps=eps)
    scale = target_rms / (rms_val + eps)
    x_norm = x * scale.unsqueeze(-1)
    x_norm = torch.clamp(x_norm, -1.0, 1.0)
    return x_norm


pth_json = '/hdd1/miseul/inf_vc/one_utterances_3s.json'
with open(pth_json, 'r') as f:
    lines = json.load(f)

pth_data  = '/hdd2/TTS_DB'
pth_abs   = '/home/miseul/gen_multi'
TARGET_SR =  16000
save_dir  = 'static/audio/src_trg'

def return_triple(model_name, epoch, k, s, c, idx, rate=16000):
    fold_name = f'outdir/{model_name}/e{epoch:03d}.ckpt/one_utterances_3s_c{c}_s{s}_None_{k}'
    tmp = lines[idx]
    spk     = tmp['spk']
    pth_src = tmp['pth_src']
    pth_trg = tmp['pth_trg']
    fname = f"{idx:03d}_{spk}"

    # ---- Load audios ----
    print(pth_src)
    src, sr_rate = sf.read(os.path.join(pth_data, pth_src))
    if sr_rate != TARGET_SR:
        src = librosa.resample(src, orig_sr=sr_rate, target_sr=TARGET_SR)

    src, index = librosa.effects.trim(src, top_db=15)
    src = torch.FloatTensor(src)
    src = normalize(src, target_rms=0.08).squeeze()
    src = src.numpy()

    trg, sr_rate = sf.read(os.path.join(pth_data, pth_trg))
    if sr_rate != TARGET_SR:
        trg = librosa.resample(trg, orig_sr=sr_rate, target_sr=TARGET_SR)

    trg, index = librosa.effects.trim(trg, top_db=15)
    trg = torch.FloatTensor(trg)
    trg = normalize(trg, target_rms=0.08).squeeze()
    trg = trg.numpy()

    os.makedirs(save_dir, exist_ok=True)
    dst = os.path.join(save_dir, fname + '_src.wav')
    sf.write(dst, src, TARGET_SR)
    dst = os.path.join(save_dir, fname + '_trg.wav')
    sf.write(dst, trg, TARGET_SR)

if __name__ == '__main__':
    model_name= '1218_infilling_0.2_beta_only'
    epoch  =482
    idx = 102

    k = 'full'
    s = 0.0
    c = 1.0
    return_triple(model_name, epoch, k, s, c, idx, rate=16000)