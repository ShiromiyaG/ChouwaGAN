"""Pitch of every clip of a dataset folder with RMVPE, at 100 frames per
second, and the filelist the trainer reads."""

import argparse
import os
import urllib.request

import librosa
import numpy as np
import torch
from tqdm import tqdm

from chouwagan.rmvpe import RMVPE0Predictor

RMVPE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
DEFAULT_RMVPE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "rmvpe.pt")


def rmvpe_weights(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading RMVPE to {path}.")
        urllib.request.urlretrieve(RMVPE_URL, path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset folder made by preprocess.py (its wavs/).")
    parser.add_argument("--rmvpe", default=DEFAULT_RMVPE, help="RMVPE weights; downloaded when missing.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.03, help="Voicing threshold.")
    args = parser.parse_args()

    wav_dir = os.path.join(args.dataset, "wavs")
    f0_dir = os.path.join(args.dataset, "f0")
    os.makedirs(f0_dir, exist_ok=True)
    names = sorted(os.listdir(wav_dir))
    if not names:
        raise SystemExit(f"No clips in {wav_dir}; run preprocess.py first.")
    model = RMVPE0Predictor(rmvpe_weights(args.rmvpe), device=args.device)
    rows = []
    for name in tqdm(names, desc="Pitch"):
        stem = os.path.splitext(name)[0]
        out = os.path.join(f0_dir, stem + ".npy")
        if not os.path.exists(out):
            audio, _ = librosa.load(os.path.join(wav_dir, name), sr=16000, mono=True)
            f0 = model.infer_from_audio(audio.astype(np.float32), thred=args.threshold)
            np.save(out, f0.astype(np.float32), allow_pickle=False)
        rows.append(f"wavs/{name}|f0/{stem}.npy")
    filelist = os.path.join(args.dataset, "filelist.txt")
    with open(filelist, "w", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")
    print(f"Wrote {filelist} ({len(rows)} clips).")


if __name__ == "__main__":
    main()
