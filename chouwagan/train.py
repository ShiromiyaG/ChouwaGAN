"""The training loop: mel, MR-STFT and harmonic losses alone for ``aux_steps``,
then against the SAN discriminator under BigVSAN's loss."""

import glob
import json
import math
import os
import random
import re
import shutil
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import soundfile as sf
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from chouwagan.data import VocoderDataset, collate, read_filelist
from chouwagan.discriminator import build_discriminator, normalize_san_weights
from chouwagan.losses import SpectralLoss, discriminator_loss, feature_loss, generator_loss, volume_augment
from chouwagan.mel import normalize_mel
from chouwagan.model import ARCHITECTURE, ChouwaGAN, EnvelopeLoss, HarmonicLoss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: The rates with a config in ``configs/<rate>.json``.
SAMPLE_RATES = (32000, 40000, 44100, 48000)
DEFAULT_SAMPLE_RATE = 44100
#: What an export keeps of the data section: the mel it renders and its normalisation.
MEL_KEYS = ("sample_rate", "hop_length", "n_fft", "win_length", "n_mels", "mel_fmin", "mel_fmax", "mel_mean", "mel_std")
#: Most of a run the spectral-only stage may take, so a short run still meets the discriminator.
AUX_MAX_SHARE = 0.25


class WeightEMA:
    """Shadow copy of the generator's weights; the running mean until its
    horizon is shorter than the run, then decay ``decay``."""

    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, 1.0 - 1.0 / self.updates)
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].lerp_(value, 1.0 - decay)
            else:
                self.shadow[key].copy_(value)

    @contextmanager
    def applied(self, model):
        """``model`` with the shadow weights in it, the live ones put back after."""
        live = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(live)

    def state_dict(self):
        return {"shadow": self.shadow, "updates": self.updates}

    def load_state_dict(self, state):
        for key, value in state["shadow"].items():
            self.shadow[key].copy_(value)
        self.updates = int(state["updates"])


def learning_rate(settings: dict, step: int, total: int) -> float:
    """Linear warmup, then a cosine from the base rate down to ``lr_final_ratio``
    of it, ``step`` of ``total``; before step 0, the warmup's start."""
    step = max(0, step)
    base = float(settings["learning_rate"])
    warmup = min(1.0, (step + 1) / max(1, int(settings.get("warmup_steps", 0))))
    final = float(settings.get("lr_final_ratio", 0.1))
    progress = min(1.0, step / max(1, total))
    return base * warmup * (final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def load_config(out_dir: str, config_path: str, sample_rate: int | None = None) -> dict:
    """The run's ``config.json``, copied on its first run from ``config_path``,
    else from the config of ``sample_rate``. A run already at another rate
    than ``sample_rate`` is refused."""
    run_config = os.path.join(out_dir, "config.json")
    if not os.path.exists(run_config):
        rate = sample_rate or DEFAULT_SAMPLE_RATE
        shutil.copyfile(config_path or os.path.join(ROOT, "configs", f"{rate}.json"), run_config)
    with open(run_config, encoding="utf-8") as handle:
        config = json.load(handle)
    own_rate = int(config["data"]["sample_rate"])
    if sample_rate and own_rate != int(sample_rate):
        raise ValueError(f"{run_config} trains at {own_rate} Hz; use another --name for {sample_rate} Hz.")
    return config


def latest(out_dir: str, prefix: str):
    paths = glob.glob(os.path.join(out_dir, f"{prefix}_*.pth"))
    steps = [(int(m.group(1)), p) for p in paths if (m := re.search(rf"{prefix}_(\d+)\.pth$", p))]
    return max(steps)[1] if steps else None


def generator_hparams(config: dict) -> dict:
    data = config["data"]
    return dict(
        sample_rate=int(data["sample_rate"]), num_mels=int(data["n_mels"]), hop_length=int(data["hop_length"]),
        **config["model"],
    )


def export(out_dir: str, name: str, config: dict, ema: WeightEMA, epoch: int, step: int) -> str:
    """The inference export: EMA weights and the settings ShiroRVC's
    ``load_vocoder`` and the Applio port read."""
    path = os.path.join(out_dir, f"{name}_vocoder_{epoch}e_{step}s.pth")
    torch.save({
        "kind": "rectified_vocoder", "architecture": ARCHITECTURE, "recipe": ARCHITECTURE,
        "config": {
            "data": {key: config["data"][key] for key in MEL_KEYS},
            "vocoder": {"model": generator_hparams(config)},
        },
        "model": {key: value.detach().float().cpu().contiguous() for key, value in ema.shadow.items()},
        "epoch": epoch, "step": step,
    }, path)
    return path


def save(out_dir, name, config, net_g, net_d, optim_g, optim_d, ema, scaler, epoch, step, finished, aux_steps):
    """G and D checkpoints of the latest step only, and an export."""
    g_path, d_path = os.path.join(out_dir, f"G_{step}.pth"), os.path.join(out_dir, f"D_{step}.pth")
    torch.save({
        "model": net_g.state_dict(), "optimizer": optim_g.state_dict(), "ema": ema.state_dict(),
        "epoch": finished, "step": step, "scaler": scaler.state_dict() if scaler is not None else None,
        "aux_steps": aux_steps,
    }, g_path)
    torch.save({"model": net_d.state_dict(), "optimizer": optim_d.state_dict(), "epoch": finished, "step": step}, d_path)
    for prefix, kept in (("G", g_path), ("D", d_path)):
        for path in glob.glob(os.path.join(out_dir, f"{prefix}_*.pth")):
            if path != kept:
                os.remove(path)
    exported = export(out_dir, name, config, ema, epoch, step)
    tqdm.write(f"Saved {os.path.basename(g_path)} and {os.path.basename(exported)}.")


def load_families(net_d, weights: dict) -> bool:
    """Load each discriminator family ``weights`` holds in the network's
    shapes; one it lacks, or holds in other shapes, keeps its fresh weights.
    True when every family loaded and the file holds no other."""
    loaded = set()
    for name, family in net_d.items():
        prefix = name + "."
        part = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
        own = family.state_dict()
        if part.keys() == own.keys() and all(part[key].shape == own[key].shape for key in own):
            family.load_state_dict(part)
            loaded.add(name)
    return loaded == set(net_d) == {key.partition(".")[0] for key in weights}


def load_pretrained(net_g, net_d, pretrained_g: str, pretrained_d: str):
    """Start from another run's weights: a G checkpoint or an export, and a D checkpoint."""
    if pretrained_g:
        state = torch.load(pretrained_g, map_location="cpu", weights_only=False)
        net_g.load_state_dict(state["model"])
    if pretrained_d:
        state = torch.load(pretrained_d, map_location="cpu", weights_only=False)
        load_families(net_d, state["model"])


def reference_clip(dataset: VocoderDataset, path: str, max_seconds: float = 10.0):
    """The preview clip as (mel, f0, audio): ``path``'s entry when given,
    else the first clip of at least two seconds, cut to ``max_seconds``."""
    order = range(len(dataset))
    if path:
        order = [i for i, (audio, _) in enumerate(dataset.entries) if os.path.abspath(audio) == os.path.abspath(path)]
        if not order:
            raise ValueError(f"{path} is not in the filelist; the preview needs its pitch file.")
    limit = int(max_seconds * dataset.sample_rate) // dataset.hop
    for index in order:
        mel, f0, audio = dataset.clip(index)
        if f0.shape[0] * dataset.hop >= 2 * dataset.sample_rate or path:
            return mel[:, :limit], f0[:limit], audio[: limit * dataset.hop]
    return None


@torch.no_grad()
def preview(net_g, ema, reference, data, device, writer, out_dir, step):
    """The reference rendered through the EMA weights, with the band contrast
    measured from it and without it (as inference without a flow that
    predicts it runs)."""
    mel, f0, audio = (value.unsqueeze(0).to(device) for value in reference)
    mel = normalize_mel(mel, data)
    renders = {}
    with ema.applied(net_g):
        net_g.eval()
        if net_g.contrast is not None:
            renders["generated"] = net_g(mel, f0, net_g.measure_contrast(audio, f0))
            renders["unmeasured"] = net_g(mel, f0)
        else:
            renders["generated"] = net_g(mel, f0)
        net_g.train()
    renders["original"] = audio
    folder = os.path.join(out_dir, "previews")
    os.makedirs(folder, exist_ok=True)
    sample_rate = data["sample_rate"]
    for kind, wave in renders.items():
        wave = wave.reshape(-1).float().clamp(-1, 1).cpu()
        writer.add_audio(f"preview/{kind}", wave.unsqueeze(0), step, sample_rate=sample_rate)
        sf.write(os.path.join(folder, f"{kind}.wav"), wave.numpy(), sample_rate)


def train(args):
    out_dir = os.path.join(args.logs, args.name)
    os.makedirs(out_dir, exist_ok=True)
    config = load_config(out_dir, args.config, args.sample_rate)
    data, settings = config["data"], config["train"]
    sample_rate, hop = int(data["sample_rate"]), int(data["hop_length"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(settings.get("seed", 1234))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    root = args.data_root if args.data_root is not None else os.path.dirname(os.path.abspath(args.filelist))
    entries = read_filelist(args.filelist, root)
    dataset = VocoderDataset(entries, data, int(settings["segment_size"]) // hop)
    if len(dataset) < args.batch_size:
        raise ValueError(f"{len(dataset)} clips is fewer than one batch of {args.batch_size}.")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate,
        drop_last=True, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )

    net_g = ChouwaGAN(**generator_hparams(config)).to(device)
    net_d = build_discriminator(config["discriminator"], sample_rate).to(device)
    adam = dict(
        betas=tuple(settings["betas"]), eps=float(settings.get("eps", 1e-8)),
        weight_decay=float(settings.get("weight_decay", 0.0)),
    )
    optim_g = torch.optim.AdamW(net_g.parameters(), float(settings["learning_rate"]), **adam)
    optim_d = torch.optim.AdamW(net_d.parameters(), float(settings["learning_rate"]), **adam)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.precision)
    scaler = torch.amp.GradScaler(device.type) if amp_dtype == torch.float16 else None

    epoch, step = 1, 0
    resumed_aux_steps = None
    g_path, d_path = latest(out_dir, "G"), latest(out_dir, "D")
    if g_path and d_path:
        state_g = torch.load(g_path, map_location="cpu", weights_only=False)
        state_d = torch.load(d_path, map_location="cpu", weights_only=False)
        net_g.load_state_dict(state_g["model"])
        optim_g.load_state_dict(state_g["optimizer"])
        # A family added or changed in the config since starts fresh, and so does D's optimizer.
        if load_families(net_d, state_d["model"]):
            optim_d.load_state_dict(state_d["optimizer"])
        else:
            print("The discriminator has families the checkpoint lacks or holds in other shapes: they and its "
                  "optimizer start fresh.")
        ema = WeightEMA(net_g, settings["ema_decay"])
        ema.load_state_dict(state_g["ema"])
        if scaler is not None and state_g.get("scaler"):
            scaler.load_state_dict(state_g["scaler"])
        epoch, step = int(state_g["epoch"]) + 1, int(state_g["step"])
        resumed_aux_steps = state_g.get("aux_steps", settings.get("aux_steps", 0))
        print(f"Resuming from {os.path.basename(g_path)} (epoch {epoch}, step {step}).")
    else:
        load_pretrained(net_g, net_d, args.pretrained_g, args.pretrained_d)
        ema = WeightEMA(net_g, settings["ema_decay"])

    spectral = SpectralLoss(config).to(device)
    c_harmonic = float(settings.get("c_harmonic", 0.0))
    harmonic = HarmonicLoss(sample_rate, hop).to(device) if c_harmonic > 0 else None
    c_envelope = float(settings.get("c_envelope", 0.0))
    envelope = None
    if c_envelope > 0:
        envelope_bands = settings.get("envelope_loss_bands_hz") or (3000, 6000, 10000, 16000)
        envelope = EnvelopeLoss(sample_rate, envelope_bands).to(device)
    contrast_dropout = float(settings.get("contrast_dropout", 0.0))
    volume_prob = float(settings.get("volume_aug_prob", 0.0))
    grad_clip = float(settings.get("grad_clip") or "inf")
    # The GAN losses as a mean over the heads instead of their sum.
    heads = net_d.heads if settings.get("average_heads", True) else 1
    total_steps = args.epochs * len(loader)
    # Kept in the checkpoint: a resume must not move the point where D joins.
    if resumed_aux_steps is not None:
        aux_steps = int(resumed_aux_steps)
    elif args.pretrained_g:
        # A pretrained G already renders, and its D would drift out of step with it.
        aux_steps = 0
    else:
        aux_steps = int(settings.get("aux_steps", 0))
        if settings.get("aux_epochs") is not None:
            # In epochs, the stage lasts as much audio whatever the batch size.
            aux_steps = round(float(settings["aux_epochs"]) * len(loader))
        aux_steps = min(aux_steps, int(AUX_MAX_SHARE * total_steps))
    log_interval = int(settings.get("log_interval", 50))
    preview_interval = int(settings.get("preview_interval", 500))
    reference = reference_clip(dataset, args.reference)
    writer = SummaryWriter(os.path.join(out_dir, "eval"))

    def autocast():
        return torch.autocast(device.type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()

    def backward(loss, optimizer, parameters):
        (loss if scaler is None else scaler.scale(loss)).backward()
        if scaler is not None:
            scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
        return norm

    print(f"{len(dataset)} clips, {len(loader)} steps per epoch, epochs {epoch} -> {args.epochs}; "
          f"G {sum(p.numel() for p in net_g.parameters()) / 1e6:.2f}M, "
          f"D {sum(p.numel() for p in net_d.parameters()) / 1e6:.2f}M ({', '.join(net_d)}); "
          + (f"spectral only until step {aux_steps}." if step < aux_steps else "with the discriminator."))
    net_g.train()
    net_d.train()
    sums, count = defaultdict(float), 0
    while epoch <= args.epochs:
        for mel, f0, y in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True):
            mel, f0, y = mel.to(device, non_blocking=True), f0.to(device, non_blocking=True), y.to(device)
            if volume_prob > 0:
                mel, y = volume_augment(mel, y, volume_prob)
            mel = normalize_mel(mel, data)
            # D's schedule starts when it joins: its own warmup, from the base rate.
            for optimizer, start in ((optim_g, 0), (optim_d, aux_steps)):
                lr = learning_rate(settings, step - start, total_steps - start)
                for group in optimizer.param_groups:
                    group["lr"] = lr
            aux_only = step < aux_steps

            conditions = ()
            if net_g.contrast is not None:
                contrast = net_g.measure_contrast(y, f0)
                if contrast_dropout > 0:
                    # Clips trained without it, as inference without it runs.
                    contrast = contrast * (torch.rand(y.shape[0], 1, 1, device=device) >= contrast_dropout)
                conditions = (contrast,)
            with autocast():
                y_hat = net_g(mel, f0, *conditions)

            losses = {}
            if not aux_only:
                with autocast():
                    real = net_d(y, san_training=True)
                    fake = net_d(y_hat.detach(), san_training=True)
                    loss_d, loss_real, loss_fake = (value / heads for value in discriminator_loss(real, fake))
                optim_d.zero_grad(set_to_none=True)
                losses["grad_norm_d"] = backward(loss_d, optim_d, net_d.parameters())
                normalize_san_weights(net_d)
                losses.update(loss_disc=loss_d, loss_disc_real=loss_real, loss_disc_fake=loss_fake)
                del real, fake

            # D only scores here; frozen, its backward still reaches y_hat.
            net_d.requires_grad_(False)
            with autocast():
                loss_spectral, loss_mel, loss_stft = spectral(y, y_hat)
                loss_g = loss_spectral
                if not aux_only:
                    d_fake = net_d(y_hat)
                    with torch.no_grad():
                        d_real = net_d(y)
                    loss_adv = generator_loss(d_fake) / heads
                    loss_fm = feature_loss(d_real, d_fake) / heads
                    loss_g = loss_g + loss_adv + loss_fm
                    losses.update(loss_adv=loss_adv, loss_fm=loss_fm)
                if harmonic is not None:
                    loss_harmonic = harmonic(y_hat, y, f0) * c_harmonic
                    loss_g = loss_g + loss_harmonic
                    losses["loss_harmonic"] = loss_harmonic
                if envelope is not None:
                    loss_envelope = envelope(y_hat, y) * c_envelope
                    loss_g = loss_g + loss_envelope
                    losses["loss_envelope"] = loss_envelope
            optim_g.zero_grad(set_to_none=True)
            losses["grad_norm_g"] = backward(loss_g, optim_g, net_g.parameters())
            if scaler is not None:
                scaler.update()
            net_d.requires_grad_(True)
            ema.update(net_g)
            step += 1

            losses.update(loss_gen_total=loss_g, loss_mel=loss_mel, loss_stft=loss_stft)
            for key, value in losses.items():
                sums[key] += float(value)
            count += 1
            if step % log_interval == 0:
                for key, value in sums.items():
                    writer.add_scalar(f"train/{key}", value / count, step)
                writer.add_scalar("train/learning_rate", optim_g.param_groups[0]["lr"], step)
                writer.add_scalar("train/learning_rate_d", optim_d.param_groups[0]["lr"], step)
                tqdm.write(f"step {step}: mel {sums['loss_mel'] / count:.3f}"
                           + (f", harmonic {sums['loss_harmonic'] / count:.3f}" if harmonic is not None else "")
                           + (f", envelope {sums['loss_envelope'] / count:.3f}" if envelope is not None else "")
                           + (f", D {sums['loss_disc'] / count:.3f}" if "loss_disc" in sums else ""))
                sums, count = defaultdict(float), 0
            if reference is not None and step % preview_interval == 0:
                preview(net_g, ema, reference, data, device, writer, out_dir, step)
            if args.save_steps and step % args.save_steps == 0:
                save(out_dir, args.name, config, net_g, net_d, optim_g, optim_d, ema, scaler, epoch, step, epoch - 1,
                     aux_steps)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save(out_dir, args.name, config, net_g, net_d, optim_g, optim_d, ema, scaler, epoch, step, epoch,
                 aux_steps)
        epoch += 1
    writer.close()
