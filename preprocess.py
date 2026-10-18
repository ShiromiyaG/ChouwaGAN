"""Cut a folder of recordings into a training set: resampled, high-passed at
48 Hz, peak-blended per recording as RVC does, sliced at silences and cut
into 3-second clips."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm

from chouwagan.slicer import Slicer

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".wma")
HIGH_PASS_HZ = 48
CLIP_SECONDS = 3.0
OVERLAP_SECONDS = 0.3
# RVC's blend toward a peak of 0.9: it keeps a quarter of the original level,
# so a quiet take stays quieter than a loud one.
MAX_AMPLITUDE = 0.9
ALPHA = 0.75


def spans_to_clips(spans, sample_rate):
    """Each voiced span cut into ``CLIP_SECONDS`` clips overlapping by ``OVERLAP_SECONDS``."""
    for span_start, span_end in spans:
        start = span_start
        while span_end - start > (CLIP_SECONDS + OVERLAP_SECONDS) * sample_rate:
            yield start, start + int(CLIP_SECONDS * sample_rate)
            start += int((CLIP_SECONDS - OVERLAP_SECONDS) * sample_rate)
        yield start, span_end


def process(job):
    index, path, out_dir, sample_rate, min_seconds, audio_format = job
    try:
        audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    except Exception as error:
        return path, 0, str(error)
    if audio.size < sample_rate * min_seconds:
        return path, 0, None
    sos = signal.butter(5, HIGH_PASS_HZ, btype="high", fs=sample_rate, output="sos")
    audio = signal.sosfiltfilt(sos, audio, padlen=min(int(6 * sample_rate / HIGH_PASS_HZ), audio.size - 1))
    peak = float(np.abs(audio).max())
    if 0 < peak <= 2.5:
        audio = audio / peak * (MAX_AMPLITUDE * ALPHA) + (1 - ALPHA) * audio
    slicer = Slicer(sr=sample_rate, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
    written = 0
    for start, end in spans_to_clips(slicer.slice_spans(audio), sample_rate):
        if end - start < sample_rate * min_seconds:
            continue
        name = f"{index}_{written}.{audio_format}"
        sf.write(os.path.join(out_dir, "wavs", name), audio[start:end].astype(np.float32), sample_rate)
        written += 1
    return path, written, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Folder of recordings, searched recursively.")
    parser.add_argument("output", help="Dataset folder; the clips go to its wavs/.")
    parser.add_argument("--sample-rate", type=int, default=44100, choices=(32000, 40000, 44100, 48000),
                        help="The rate of the config the set will train with.")
    parser.add_argument("--min-seconds", type=float, default=0.5, help="Shorter clips are dropped.")
    parser.add_argument("--format", default="flac", choices=("flac", "wav"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()

    paths = sorted(
        os.path.join(folder, name)
        for folder, _, names in os.walk(args.input) for name in names
        if name.lower().endswith(AUDIO_EXTENSIONS)
    )
    if not paths:
        raise SystemExit(f"No audio under {args.input}.")
    os.makedirs(os.path.join(args.output, "wavs"), exist_ok=True)
    jobs = [(i, p, args.output, args.sample_rate, args.min_seconds, args.format) for i, p in enumerate(paths)]
    clips = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for path, written, error in tqdm(pool.map(process, jobs), total=len(jobs), desc="Preprocess"):
            if error:
                tqdm.write(f"Skipped {path}: {error}")
            clips += written
    print(f"{clips} clips from {len(paths)} recordings in {os.path.join(args.output, 'wavs')}.")


if __name__ == "__main__":
    main()
