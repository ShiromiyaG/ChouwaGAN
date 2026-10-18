"""Resynthesise recordings through a ChouwaGAN export: each file's log mel and
RMVPE pitch rendered back to audio, as ShiroRVC renders a flow's mel, with
the high-band diffuser after the vocoder."""

import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from chouwagan.data import f0_to_mel_rate, mel_frames
from chouwagan.inference import load_export, render
from chouwagan.mel import LogMel, normalize_mel
from chouwagan.rmvpe import RMVPE0Predictor
from extract import DEFAULT_RMVPE, rmvpe_weights
from preprocess import AUDIO_EXTENSIONS

#: RMVPE's input rate.
PITCH_RATE = 16000


def audio_files(paths):
    """The files of ``paths``, and the audio files under any folder among them."""
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in sorted(os.walk(path)):
                yield from (os.path.join(root, n) for n in sorted(names) if n.lower().endswith(AUDIO_EXTENSIONS))
        else:
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Audio files or folders.")
    parser.add_argument("--export", required=True, help="The vocoder export (.pth).")
    parser.add_argument("--output", default="outputs", help="Folder for the results.")
    parser.add_argument("--contrast", choices=("measured", "none"), default="measured",
                        help="Band contrast measured from each input, or left out as without a flow to predict it.")
    parser.add_argument("--no-diffuse", action="store_true", help="Leave the high-band diffuser out.")
    parser.add_argument("--rmvpe", default=DEFAULT_RMVPE, help="RMVPE weights; downloaded when missing.")
    parser.add_argument("--threshold", type=float, default=0.03, help="Voicing threshold.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    vocoder = load_export(args.export, diffuse=not args.no_diffuse).to(device)
    data = vocoder.data
    sample_rate, hop = int(data["sample_rate"]), int(data["hop_length"])
    mel_fn = LogMel.from_config(data).to(device)
    pitch = RMVPE0Predictor(rmvpe_weights(args.rmvpe), device=str(device))
    os.makedirs(args.output, exist_ok=True)
    for path in tqdm(list(audio_files(args.inputs)), desc="Rendering"):
        audio, _ = librosa.load(path, sr=sample_rate, mono=True)
        low = librosa.resample(audio, orig_sr=sample_rate, target_sr=PITCH_RATE)
        f0 = pitch.infer_from_audio(low.astype(np.float32), thred=args.threshold)
        frames = min(audio.shape[0] // hop, mel_frames(f0.shape[0], sample_rate, hop))
        wave = torch.from_numpy(audio[:frames * hop]).float().unsqueeze(0).to(device)
        mel = normalize_mel(mel_fn(wave)[:, :, :frames], data)
        f0 = f0_to_mel_rate(torch.from_numpy(f0.astype(np.float32)), frames, sample_rate, hop).unsqueeze(0).to(device)
        out = render(vocoder, mel, f0, audio=wave if args.contrast == "measured" else None)
        name = os.path.splitext(os.path.basename(path))[0] + "_chouwagan.wav"
        sf.write(os.path.join(args.output, name), out[0].clamp(-1, 1).cpu().numpy(), sample_rate)
    print(f"Wrote the results to {args.output}.")


if __name__ == "__main__":
    main()
