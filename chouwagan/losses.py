import math

import torch
from torch import nn
from torch.nn import functional as F

from chouwagan.mel import LogMel


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral convergence and log-magnitude losses, averaged over the resolutions."""

    def __init__(self, fft_sizes, hop_sizes, win_lengths):
        super().__init__()
        self.resolutions = list(zip(fft_sizes, hop_sizes, win_lengths))
        for index, (_, _, win_length) in enumerate(self.resolutions):
            self.register_buffer(f"window_{index}", torch.hann_window(win_length), persistent=False)

    def _magnitude(self, x, index):
        fft_size, hop_size, win_length = self.resolutions[index]
        window = getattr(self, f"window_{index}")
        spec = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
        return torch.clamp(spec.abs(), min=10**-3.5)

    def forward(self, x, y):
        """Predicted ``x`` and target ``y``, [B, T] each."""
        x, y = x.float(), y.float()
        sc_loss = mag_loss = 0.0
        for index in range(len(self.resolutions)):
            x_mag, y_mag = self._magnitude(x, index), self._magnitude(y, index)
            sc_loss = sc_loss + torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
            mag_loss = mag_loss + F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        return sc_loss / len(self.resolutions), mag_loss / len(self.resolutions)


class SpectralLoss(nn.Module):
    """Mel L1 to Nyquist at ``c_mel`` plus MR-STFT at ``c_stft``: the mel's wide
    high bands cannot tell a harmonic from noise, the linear bins can."""

    def __init__(self, config: dict):
        super().__init__()
        data, settings = config["data"], config["train"]
        self.mel = LogMel.from_config(data, fmax=data["sample_rate"] / 2)
        self.stft = MultiResolutionSTFTLoss(
            settings["loss_fft_sizes"], settings["loss_hop_sizes"], settings["loss_win_lengths"]
        )
        self.c_mel = float(settings["c_mel"])
        self.c_stft = float(settings["c_stft"])

    def forward(self, y, y_hat):
        """Returns the total and its mel and MR-STFT terms."""
        y, y_hat = y.squeeze(1).float(), y_hat.squeeze(1).float()
        loss_mel = F.l1_loss(self.mel(y_hat), self.mel(y)) * self.c_mel
        sc_loss, mag_loss = self.stft(y_hat, y)
        loss_stft = (sc_loss + mag_loss) * self.c_stft
        return loss_mel + loss_stft, loss_mel, loss_stft


def _heads(outputs: dict):
    return [d for heads, _ in outputs.values() for d in heads]


def discriminator_loss(real: dict, fake: dict):
    """BigVSAN's loss over every SAN head, and its real and fake halves:
    softplus² on the function output, and on the direction output the real
    term against minus ``softplus(1 - d)²`` on the fake one."""
    loss_real = loss_fake = 0.0
    for (real_fun, real_dir), (fake_fun, fake_dir) in zip(_heads(real), _heads(fake)):
        loss_real = loss_real + torch.mean(F.softplus(1 - real_fun.float()) ** 2) + torch.mean(
            F.softplus(1 - real_dir.float()) ** 2
        )
        loss_fake = loss_fake + torch.mean(F.softplus(fake_fun.float()) ** 2) - torch.mean(
            F.softplus(1 - fake_dir.float()) ** 2
        )
    return loss_real + loss_fake, loss_real, loss_fake


def generator_loss(fake: dict):
    """The softplus loss SAN heads pair with, over every head."""
    return sum(torch.mean(F.softplus(1.0 - d.float()).square()) for d in _heads(fake))


def feature_loss(real: dict, fake: dict):
    """Feature matching over every layer of every head, x2."""
    loss = 0.0
    for name in fake:
        for head_real, head_fake in zip(real[name][1], fake[name][1]):
            for r, g in zip(head_real, head_fake):
                loss = loss + torch.mean(torch.abs(r.float() - g.float()))
    return loss * 2


def volume_augment(mel, audio, probability: float):
    """SingingVocoders' volume augmentation: with ``probability`` per clip, a
    gain of e^-3 up to e^3 that keeps the peak under 1, on the audio and, as a
    shift, on its log mel."""
    peak = audio.abs().amax(dim=(1, 2)) + 1e-5
    high = torch.clamp(torch.log(1 / peak), max=3.0)
    shift = -3.0 + (high + 3.0) * torch.rand_like(high)
    shift = torch.where(torch.rand_like(high) < probability, shift, torch.zeros_like(shift))
    mel = torch.clamp(mel + shift.view(-1, 1, 1), min=math.log(1e-5))
    return mel, audio * shift.exp().view(-1, 1, 1)
