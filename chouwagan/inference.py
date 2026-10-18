"""Rendering with an export outside training: the export loader, the high-band
diffuser that inference puts after the vocoder, and a render in passes for
long inputs."""

import math

import torch
from torch import nn

from chouwagan.model import ARCHITECTURE, SOURCE_F0_FLOOR, ChouwaGAN

#: The diffuser's bins turn from here up, faded in over the kHz below.
DIFFUSE_HZ = 4000.0
DIFFUSE_SEED = 0
#: Frames per pass, the context rendered on each side and the crossfade at
#: each join: every pass starts its pulse train at phase zero, so a hard join
#: would click.
CHUNK = 6000
MIN_CHUNK = 375
#: Frames per pass while measuring the band contrast, whose harmonic grid
#: takes far more memory than the render.
MEASURED_CHUNK = 1500
CONTEXT = 50
CROSSFADE = 4


class Diffuser(nn.Module):
    """[B, 1, T] and the f0 [B, frames] it was rendered from -> [B, 1, T].
    Above ``DIFFUSE_HZ``, each bin of a ``n_fft`` STFT turns by a fixed random
    phase of the harmonic of f0 nearest to it, so each harmonic turns whole
    and keeps its phase as it moves. The pulse train has every harmonic in
    phase, which up there sounds like static; scattered harmonic phases spread
    each pulse's top over the cycle and leave the harmonics sharp, where a
    phase per bin blurred them. A generator trained through either learned
    steady lines, so it runs at inference only."""

    def __init__(self, sample_rate, n_fft, hop, frame_hop):
        super().__init__()
        self.n_fft, self.hop, self.frame_hop = int(n_fft), int(hop), int(frame_hop)
        bins = torch.linspace(0, sample_rate / 2, self.n_fft // 2 + 1).view(1, -1, 1)
        generator = torch.Generator().manual_seed(DIFFUSE_SEED)
        phases = math.pi * (2 * torch.rand(int(sample_rate / 2 / SOURCE_F0_FLOOR) + 2, generator=generator) - 1)
        self.register_buffer("bins", bins, persistent=False)
        self.register_buffer("ramp", ((bins - DIFFUSE_HZ) / 1000 + 1).clamp(0, 1), persistent=False)
        self.register_buffer("phases", phases, persistent=False)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    def forward(self, audio, f0):
        x, window = audio.squeeze(1).float(), self.window.float()
        spectrum = torch.stft(x, self.n_fft, self.hop, window=window, return_complex=True)
        frame = torch.arange(spectrum.shape[-1], device=x.device) * self.hop // self.frame_hop
        f0 = f0.float()[:, frame.clamp(max=f0.shape[-1] - 1)].unsqueeze(1)
        harmonic = torch.round(self.bins / f0.clamp(min=SOURCE_F0_FLOOR)).long().clamp(max=len(self.phases) - 1)
        phase = self.phases[harmonic] * self.ramp * (f0 > 0)
        spectrum = spectrum * torch.polar(torch.ones_like(phase), phase)
        x = torch.istft(spectrum, self.n_fft, self.hop, window=window, length=x.shape[-1])
        return x.unsqueeze(1).to(audio.dtype)


class Vocoder(nn.Module):
    """An export's generator and its mel settings (``data``), with a
    ``Diffuser`` on the output when ``diffuse``. ``forward`` takes what the
    generator does: the normalised mel [B, n_mels, T], f0 [B, T] and,
    optionally, the contrast input [B, bands + 1, T]."""

    def __init__(self, generator: ChouwaGAN, data: dict, diffuse: bool = True):
        super().__init__()
        self.generator = generator
        self.data = data
        self.diffuser = None
        if diffuse:
            self.diffuser = Diffuser(
                generator.sample_rate, generator.noise_fft, generator.synthesis_hop, generator.hop_length
            )

    def forward(self, mel, f0, contrast=None):
        audio = self.generator(mel, f0, contrast)
        return audio if self.diffuser is None else self.diffuser(audio, f0)


def load_export(path: str, diffuse: bool = True) -> Vocoder:
    """The export at ``path`` as a ``Vocoder`` in eval mode, on the CPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "rectified_vocoder" or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(f"{path} is not a ChouwaGAN export.")
    generator = ChouwaGAN(**checkpoint["config"]["vocoder"]["model"])
    generator.load_state_dict(checkpoint["model"])
    return Vocoder(generator, checkpoint["config"]["data"], diffuse).eval()


@torch.no_grad()
def render(vocoder: Vocoder, mel, f0, audio=None, chunk: int | None = None):
    """[1, samples] from a normalised mel [1, n_mels, T] and f0 [1, T], in
    passes of ``chunk`` frames crossfaded at the joins; the passes halve when
    one does not fit in memory. With ``audio`` [1, T * hop] and a generator
    that reads the band contrast, each pass measures it from its own stretch
    of ``audio``, which over a whole song at once would not fit."""
    hop = int(vocoder.data["hop_length"])
    frames = mel.shape[-1]
    half = CROSSFADE * hop // 2
    measure = audio is not None and vocoder.generator.contrast is not None
    chunk = chunk or (MEASURED_CHUNK if measure else CHUNK)
    while True:
        output = mel.new_zeros(frames * hop, dtype=torch.float32)
        try:
            for start in range(0, frames, chunk):
                stop = min(frames, start + chunk)
                left, right = max(0, start - CONTEXT), min(frames, stop + CONTEXT)
                pitch = f0[:, left:right]
                contrast = None
                if measure:
                    contrast = vocoder.generator.measure_contrast(audio[:, left * hop:right * hop], pitch)
                part = vocoder(mel[..., left:right], pitch, contrast)[0, 0].float()
                position = torch.arange(left * hop, left * hop + part.shape[0], device=part.device)
                weight = torch.ones_like(part)
                if start > 0:
                    weight = weight * ((position - (start * hop - half)) / (2 * half)).clamp(0, 1)
                if stop < frames:
                    weight = weight * (((stop * hop + half) - position) / (2 * half)).clamp(0, 1)
                output[left * hop:left * hop + part.shape[0]] += part * weight
            return output.unsqueeze(0)
        except torch.cuda.OutOfMemoryError:
            if chunk <= MIN_CHUNK:
                raise
        # Outside the handler, whose traceback holds the pass's tensors.
        del output
        torch.cuda.empty_cache()
        chunk = max(MIN_CHUNK, chunk // 2)
        print(f"Out of memory; rendering {chunk} frames per pass.")
