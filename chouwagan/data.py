import os
import random

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from chouwagan.mel import LogMel

#: Frames per second of the pitch files, RMVPE's at 16 kHz with a hop of 160.
F0_RATE = 100


def read_filelist(path: str, root: str = "") -> list:
    """(audio, f0) pairs of a filelist: ``audio|f0`` rows, or an RVC filelist
    (``audio|features|f0|f0_voiced|speaker``), whose fourth field is the
    pitch in Hz. Relative paths start at ``root``."""
    entries = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split("|")
            if len(fields) < 2:
                continue
            audio, f0 = (fields[0], fields[3]) if len(fields) >= 5 else (fields[0], fields[1])
            entries.append(tuple(p if os.path.isabs(p) else os.path.join(root, p) for p in (audio, f0)))
    return entries


def mel_frames(f0_frames: int, sample_rate: int, hop: int) -> int:
    """Mel frames covered by ``f0_frames`` pitch frames."""
    return int(f0_frames * sample_rate / (hop * F0_RATE))


def f0_to_mel_rate(f0: torch.Tensor, frames: int, sample_rate: int, hop: int) -> torch.Tensor:
    """Pitch [time] at ``F0_RATE`` -> ``frames`` mel frames, interpolated only
    between two voiced frames; elsewhere the nearer one, so no frame gets a
    pitch halfway to zero."""
    position = torch.arange(frames, dtype=torch.float64)
    position = (position * hop * F0_RATE / sample_rate).clamp(max=f0.shape[-1] - 1)
    left = position.floor().long()
    right = (left + 1).clamp(max=f0.shape[-1] - 1)
    weight = (position - left).float()
    a, b = f0[left], f0[right]
    nearest = torch.where(weight < 0.5, a, b)
    return torch.where((a > 0) & (b > 0), a * (1 - weight) + b * weight, nearest)


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != sample_rate:
        raise ValueError(f"{path} is {rate} Hz; the config trains at {sample_rate} Hz. Run preprocess.py.")
    return torch.from_numpy(audio.mean(-1))


class VocoderDataset(Dataset):
    """(mel, f0, audio) crops of ``segment_frames`` frames; the mel is taken
    over the whole clip before cropping, as at inference. Without
    ``augment`` the crop is the middle."""

    def __init__(self, entries, data: dict, segment_frames: int, augment: bool = True):
        self.entries = entries
        self.data = data
        self.sample_rate = int(data["sample_rate"])
        self.hop = int(data["hop_length"])
        self.segment_frames = int(segment_frames)
        self.augment = augment
        self.mel = LogMel.from_config(data)

    def __len__(self):
        return len(self.entries)

    def clip(self, index):
        """Clip ``index`` whole: its mel [n_mels, T], f0 [T] and audio [T * hop]."""
        audio_path, f0_path = self.entries[index]
        audio = load_audio(audio_path, self.sample_rate)
        f0 = torch.from_numpy(np.load(f0_path, allow_pickle=False).astype(np.float32))
        frames = min(audio.shape[0] // self.hop, mel_frames(f0.shape[0], self.sample_rate, self.hop))
        f0 = f0_to_mel_rate(f0, frames, self.sample_rate, self.hop)
        audio = audio[: frames * self.hop]
        with torch.no_grad():
            mel = self.mel(audio.unsqueeze(0))[0, :, :frames]
        return mel, f0, audio

    def __getitem__(self, index):
        mel, f0, audio = self.clip(index)
        frames = f0.shape[0]
        if frames < self.segment_frames:
            short = self.segment_frames - frames
            audio = F.pad(audio, (0, short * self.hop))
            f0 = F.pad(f0, (0, short))
            # Silence under the floor, as the mel of the padding reads.
            mel = F.pad(mel, (0, short), value=float(np.log(1e-5)))
            frames = self.segment_frames
        start = random.randint(0, frames - self.segment_frames) if self.augment else (frames - self.segment_frames) // 2
        stop = start + self.segment_frames
        return mel[:, start:stop], f0[start:stop], audio[start * self.hop : stop * self.hop]


def collate(batch):
    mel, f0, audio = zip(*batch)
    return torch.stack(mel), torch.stack(f0), torch.stack(audio).unsqueeze(1)
