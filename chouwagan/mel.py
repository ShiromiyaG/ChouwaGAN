import torch
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import functional as F


class LogMel(nn.Module):
    """Log mel spectrogram at one frame per ``hop_length`` samples, the one
    ShiroRVC's rectified flow produces: SingingVocoders' framing, natural log
    over a floor of 1e-5."""

    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels, fmin, fmax):
        super().__init__()
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        basis = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.register_buffer("basis", torch.from_numpy(basis).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    @classmethod
    def from_config(cls, data: dict, fmax=None) -> "LogMel":
        return cls(
            data["sample_rate"], data["n_fft"], data["win_length"], data["hop_length"], data["n_mels"],
            data["mel_fmin"], data["mel_fmax"] if fmax is None else fmax,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """``audio`` [batch, samples] -> [batch, n_mels, samples // hop]."""
        pad = self.win_length - self.hop_length
        # Reflection needs more samples than it pads.
        short = (pad + 1) // 2 + 1 - audio.shape[-1]
        if short > 0:
            audio = F.pad(audio, (0, short))
        audio = F.pad(audio.float().unsqueeze(1), (pad // 2, (pad + 1) // 2), mode="reflect").squeeze(1)
        spec = torch.stft(
            audio, self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window,
            center=False, return_complex=True,
        ).abs()
        # Out of autocast: in BF16 the log mel would come in steps of ~0.03.
        with torch.autocast(spec.device.type, enabled=False):
            return torch.log(torch.clamp(self.basis @ spec.float(), min=1e-5))


def normalize_mel(mel: torch.Tensor, data: dict) -> torch.Tensor:
    return (mel - data["mel_mean"]) / data["mel_std"]
