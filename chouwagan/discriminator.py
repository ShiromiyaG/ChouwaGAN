"""ChouwaGAN's discriminator: compressed multi-resolution spectrograms, narrow
period heads, the high bands' envelopes and how their phase moves, each ending
in a SAN projection (Takida et al., "SAN: Inducing Metrizability of GAN with
Discriminative Normalized Linear Layer")."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

#: Floor under a compressed magnitude, as a share of a full-scale sine's:
#: bounds the slope at silence.
COMPRESSION_FLOOR = 1e-4
#: Of the phase heads' unit-RMS band: where a bin's phase starts to weigh.
PHASE_FLOOR = 0.2


def normalize_weight(weight):
    """Each output channel's kernel on the unit sphere."""
    shape = (weight.shape[0],) + (1,) * (weight.ndim - 1)
    return weight / weight.flatten(1).norm(p=2, dim=1).clamp_min(1e-12).view(shape)


class SANConv2d(nn.Conv2d):
    """A SAN head: a unit-norm direction times a bounded scale. In the
    discriminator update (``san_training``) it gives the (function, direction)
    pair: the function trains the trunk and the scale against a detached
    direction, the direction only itself from a detached input."""

    SCALE_MIN = 1e-4
    SCALE_MAX = 4.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scale = self.weight.detach().flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
        self.weight = nn.Parameter(self.weight.detach() / scale.view(-1, 1, 1, 1))
        self.scale = nn.Parameter(scale)
        if self.bias is not None:
            # On the input side: an additive term after the projection would
            # leave the geometry the method rests on.
            self.bias = nn.Parameter(torch.zeros(self.in_channels, dtype=self.weight.dtype))

    def forward(self, x, san_training=False):
        def conv(inputs, weight):
            return F.conv2d(inputs, weight, None, self.stride, self.padding, self.dilation, self.groups)

        # In FP32: a normalised projection in half precision loses the small
        # differences it exists to measure.
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            direction = normalize_weight(self.weight.float())
            scale = self.scale.float().clamp(self.SCALE_MIN, self.SCALE_MAX).view(1, -1, 1, 1)
            if self.bias is not None:
                x = x + self.bias.float().view(1, -1, 1, 1)
            if not san_training:
                return conv(x, direction) * scale
            return conv(x, direction.detach()) * scale, conv(x.detach(), direction) * scale.detach()

    @torch.no_grad()
    def normalize_weight(self):
        self.weight.copy_(normalize_weight(self.weight))


def normalize_san_weights(net):
    """Put every SAN direction back on the sphere, after each optimizer step."""
    for module in net.modules():
        if isinstance(module, SANConv2d):
            module.normalize_weight()


def _output(head, x, san_training, flatten=False):
    """The head's last conv; a SAN head in the discriminator update gives the
    (function, direction) pair."""
    shape = (lambda y: torch.flatten(y, 1, -1)) if flatten else (lambda y: y)
    if not head.use_san:
        return shape(head.output_conv(x))
    out = head.output_conv(x, san_training=san_training)
    return tuple(shape(y) for y in out) if san_training else shape(out)


class PeriodDiscriminator(nn.Module):
    """HiFi-GAN's period discriminator, as Wavehax sizes it."""

    def __init__(self, period, channels=32, kernel_sizes=(5, 3), downsample_scales=(3, 3, 3, 3, 1),
                 max_channels=1024, use_san=False):
        super().__init__()
        self.period = period
        self.use_san = bool(use_san)
        self.convs = nn.ModuleList()
        in_channels, out_channels = 1, channels
        for scale in downsample_scales:
            self.convs.append(weight_norm(nn.Conv2d(
                in_channels, out_channels, (kernel_sizes[0], 1), (scale, 1), ((kernel_sizes[0] - 1) // 2, 0)
            )))
            in_channels, out_channels = out_channels, min(out_channels * 4, max_channels)
        output = dict(kernel_size=(kernel_sizes[1], 1), padding=((kernel_sizes[1] - 1) // 2, 0))
        if self.use_san:
            self.output_conv = SANConv2d(out_channels, 1, **output)
        else:
            self.output_conv = weight_norm(nn.Conv2d(out_channels, 1, **output))

    def forward(self, x, san_training=False):
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - t % self.period
            x = F.pad(x, (0, pad), "reflect")
            t += pad
        x = x.view(b, c, t // self.period, self.period)
        fmap = []
        for index, conv in enumerate(self.convs):
            if index:
                x = conv(x)
            else:
                # On the waveform, out of autocast: BF16 would keep 8 bits of each sample.
                with torch.autocast(x.device.type, enabled=False):
                    x = conv(x.float())
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        return _output(self, x, san_training, flatten=True), fmap


class SpectralDiscriminator(nn.Module):
    """UnivNet's spectrogram discriminator, as Wavehax sizes it.

    ``compression`` under 1 raises the magnitude to that power, which brings
    the quiet high band up to the scale of the low one; ``floor`` is added
    first and bounds the slope on quiet bins. ``first_channels`` narrows the
    input and the first conv, which runs at the full resolution and costs
    most of the head. With ``phase``, from bin ``min_bin`` up, the input is
    how the phase moves: see ``_phase_relations``."""

    KERNEL_SIZES = ((7, 5), (5, 3), (5, 3), (3, 3), (3, 3), (3, 3))
    STRIDES = ((2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (1, 1))

    def __init__(self, fft_size, hop_size, win_length, channels=32, compression=1.0, floor=COMPRESSION_FLOOR,
                 use_san=False, first_channels=None, phase=False, min_bin=0):
        super().__init__()
        self.use_san = bool(use_san)
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.compression = float(compression)
        self.floor = float(floor)
        self.phase = bool(phase)
        self.min_bin = int(min_bin)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        first = int(first_channels or channels)
        self.input_conv = weight_norm(nn.Conv2d(5 if self.phase else 1, first, 1))
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(
                first if index == 0 else channels, channels, kernel,
                padding=(kernel[0] // 2, kernel[1] // 2), stride=stride,
            ))
            for index, (kernel, stride) in enumerate(zip(self.KERNEL_SIZES, self.STRIDES))
        )
        self.output_conv = SANConv2d(channels, 1, 1) if self.use_san else weight_norm(nn.Conv2d(channels, 1, 1))

    def forward(self, x, san_training=False):
        x = torch.stft(
            x.squeeze(1).float(), self.fft_size, self.hop_size, self.win_length, self.window,
            center=True, return_complex=True,
        )[:, self.min_bin:]
        if self.phase:
            x = self._phase_relations(x)
        else:
            x = x.abs().unsqueeze(1)
            if self.compression != 1.0:
                x = (x / (0.5 * self.win_length) + self.floor) ** self.compression
        # On the magnitude, out of autocast: unnormalised, FP16's range is
        # short for its gradient.
        with torch.autocast(x.device.type, enabled=False):
            x = F.leaky_relu(self.input_conv(x.float()), 0.1)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        return _output(self, x, san_training), fmap

    def _phase_relations(self, spec):
        """[B, 5, bins, frames] from a [B, bins, frames] STFT: the compressed
        magnitude, and as phasors it weights, the phase advance over a hop
        less the bin's own and the phase step from the bin below. Raw real
        and imaginary parts turn by tens of radians a frame up here, which
        convolutions barely read; these relations they do. The band is first
        brought to unit RMS, its level being the magnitude heads' to judge."""
        spec = spec / (spec.abs().square().mean((1, 2), keepdim=True).sqrt() + 1e-8)
        magnitude = (spec.abs() + PHASE_FLOOR) ** self.compression - PHASE_FLOOR ** self.compression
        bins = torch.arange(self.min_bin, self.min_bin + spec.shape[1], device=spec.device)
        advance_angle = -2 * math.pi * bins * self.hop_size / self.fft_size
        own = torch.polar(torch.ones(bins.shape, device=spec.device), advance_angle)
        advance = spec[..., 1:] * spec[..., :-1].conj() * own.view(1, -1, 1)
        step = spec[:, 1:] * spec[:, :-1].conj()
        # Over a floor, so a quiet bin's phasor shrinks and its slope stays bounded.
        advance = advance / (advance.abs() + PHASE_FLOOR ** 2) * magnitude[..., 1:]
        step = step / (step.abs() + PHASE_FLOOR ** 2) * magnitude[:, 1:]
        return torch.cat((
            magnitude.unsqueeze(1),
            F.pad(torch.view_as_real(advance).permute(0, 3, 1, 2), (1, 0)),
            F.pad(torch.view_as_real(step).permute(0, 3, 1, 2), (0, 0, 1, 0)),
        ), 1)


class Heads(nn.Module):
    """A family of discriminators: ``(outputs, feature maps)``, one per head."""

    def __init__(self, discriminators):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y, san_training=False):
        outputs, fmaps = [], []
        for discriminator in self.discriminators:
            output, fmap = discriminator(y, san_training)
            outputs.append(output)
            fmaps.append(fmap)
        return outputs, fmaps


class EnvelopeDiscriminator(nn.Module):
    """How the loudness of each band between ``band_edges_hz`` moves within
    and across glottal cycles: its Hilbert envelope in log, less its mean
    over the clip (the level is the spectral losses'), at 1 / ``downsample``
    of the rate, through narrow strided convs. Magnitude only: on the raw
    waveform the top is ~40 dB under the low band, and a multi-band view with
    phase was seen to kill the generator."""

    def __init__(self, sample_rate, band_edges_hz, downsample=4, channels=(32, 64, 128, 128, 128),
                 strides=(2, 2, 2, 2, 1), kernel_size=9, ramp_hz=500.0, use_san=False):
        super().__init__()
        self.use_san = bool(use_san)
        self.sample_rate = float(sample_rate)
        self.band_edges = sorted(float(hz) for hz in band_edges_hz)
        self.downsample = int(downsample)
        self.ramp_hz = float(ramp_hz)
        self.convs = nn.ModuleList()
        in_channels = len(self.band_edges) - 1
        for out_channels, stride in zip(channels, strides):
            self.convs.append(weight_norm(nn.Conv2d(
                in_channels, out_channels, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0)
            )))
            in_channels = out_channels
        output = dict(kernel_size=(3, 1), padding=(1, 0))
        if self.use_san:
            self.output_conv = SANConv2d(in_channels, 1, **output)
        else:
            self.output_conv = weight_norm(nn.Conv2d(in_channels, 1, **output))

    def _masks(self, freqs):
        """[bands, bins]: each band with linear ramps ``ramp_hz`` wide at its
        edges, none at Nyquist."""
        nyquist = self.sample_rate / 2
        masks = []
        for low, high in zip(self.band_edges[:-1], self.band_edges[1:]):
            mask = ((freqs - low) / self.ramp_hz + 0.5).clamp(0, 1)
            if high < nyquist:
                mask = mask * ((high - freqs) / self.ramp_hz + 0.5).clamp(0, 1)
            masks.append(mask)
        return torch.stack(masks)

    def forward(self, x, san_training=False):
        # The transforms and the first conv in FP32, as the period heads' first.
        with torch.autocast(x.device.type, enabled=False):
            x = x.squeeze(1).float()
            samples = x.shape[-1]
            freqs = torch.fft.rfftfreq(samples, 1.0 / self.sample_rate, device=x.device)
            # The analytic signal: positive frequencies doubled, the rest zero.
            spectrum = torch.fft.rfft(x).unsqueeze(1) * (2 * self._masks(freqs))
            analytic = torch.fft.ifft(spectrum, n=samples)
            envelope = (analytic.real.square() + analytic.imag.square() + 1e-12).sqrt()
            envelope = F.avg_pool1d(envelope, self.downsample)
            log = torch.log(envelope + 1e-3 * envelope.mean(-1, keepdim=True) + 1e-8)
            x = F.leaky_relu(self.convs[0]((log - log.mean(-1, keepdim=True)).unsqueeze(-1)), 0.1)
        fmap = [x]
        for conv in self.convs[1:]:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        return _output(self, x, san_training, flatten=True), fmap


class Discriminator(nn.ModuleDict):
    """The families by name; ``forward`` gives ``{family: (outputs, feature maps)}``."""

    def forward(self, y, san_training=False):
        return {name: family(y, san_training) for name, family in self.items()}

    @property
    def heads(self) -> int:
        return sum(len(family.discriminators) for family in self.values())


def build_discriminator(settings: dict, sample_rate: int) -> Discriminator:
    """The compressed MRD, narrowed at the input by ``mrd_first_channels``; if
    ``periods`` names any, narrow period heads; with ``envelope_bands_hz``,
    the envelope head; with ``phase_fft_sizes``, phase heads above
    ``phase_min_hz``; all ending in SAN projections with ``use_san``.
    ``mrd_floor`` bounds the compression's slope, which otherwise sends most
    of the MRD's gradient to quiet bins."""
    san = bool(settings.get("use_san", False))
    spectral = dict(
        compression=settings.get("mrd_compression", 0.3),
        floor=settings.get("mrd_floor", COMPRESSION_FLOOR),
        first_channels=settings.get("mrd_first_channels"),
        use_san=san,
    )
    families = {"mrd": Heads(
        SpectralDiscriminator(*sizes, **spectral)
        for sizes in zip(settings["mrd_fft_sizes"], settings["mrd_hop_sizes"], settings["mrd_win_lengths"])
    )}
    if settings.get("periods"):
        families["mpd"] = Heads(
            PeriodDiscriminator(p, max_channels=int(settings.get("mpd_max_channels", 256)), use_san=san)
            for p in settings["periods"]
        )
    if settings.get("envelope_bands_hz"):
        families["env"] = Heads([EnvelopeDiscriminator(
            sample_rate, settings["envelope_bands_hz"], int(settings.get("envelope_downsample", 4)), use_san=san
        )])
    if settings.get("phase_fft_sizes"):
        # The magnitude views cannot see how the high harmonics line up in time.
        low = float(settings.get("phase_min_hz", 0.0))
        families["phase"] = Heads(
            SpectralDiscriminator(*sizes, phase=True, min_bin=math.ceil(low * sizes[0] / sample_rate), **spectral)
            for sizes in zip(settings["phase_fft_sizes"], settings["phase_hop_sizes"], settings["phase_win_lengths"])
        )
    return Discriminator(families)
