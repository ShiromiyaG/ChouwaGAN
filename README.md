<p align="center">
  <img src="assets/logo.png" alt="ChouwaGAN" width="600">
</p>

A mel vocoder whose network predicts only coarse gains. A frame-rate ConvNeXt
reads a normalised log mel and f0 and predicts, per band, the gains of two
analytic sources:

- a phase-aligned, band-limited pulse train with every harmonic of f0 to
  Nyquist, under a mixed-phase gain (minimum phase for the tract, maximum
  phase for the glottis);
- noise that follows the glottal cycle as a pulse of predicted depth and
  sharpness per frequency region.

Both are synthesised by overlap-add of short-time spectra at a quarter of the
mel hop, so no layer runs at the audio rate.

Optionally the network also reads the harmonic contrast of a few bands
(`contrast_band_edges`), which a 128-bin mel cannot show above a few kHz. It is
measured from the audio in training and predicted by ShiroRVC's rectified flow
at inference; left out, the vocoder renders from the mel alone.

It renders the mel of ShiroRVC's rectified flow: 128 bins from 0 Hz to Nyquist,
natural log over a floor of 1e-5.

| Rate     | Hop         | n_fft | Contrast bands up to |
|----------|-------------|-------|----------------------|
| 32 kHz   | 320 (10 ms) | 1536  | 12 kHz               |
| 40 kHz   | 400 (10 ms) | 2048  | 17 kHz               |
| 44.1 kHz | 512         | 2048  | 17 kHz               |
| 48 kHz   | 480 (10 ms) | 2048  | 20 kHz               |

`configs/<rate>.json` holds the mel, the model, the discriminator and the
training settings of each rate. The exports load in ShiroRVC, and at 44.1 kHz
in the Applio port.

## Layout

```
preprocess.py       recordings -> resampled, levelled 3 s clips
extract.py          clips -> RMVPE pitch + filelist.txt
train.py            training entry point
infer.py            recordings -> mel + pitch -> audio through an export
configs/            one config per sample rate
chouwagan/
  model.py          generator, HarmonicLoss, band contrast
  discriminator.py  MRD, MPD and envelope discriminators (SAN heads)
  losses.py         mel, MR-STFT and adversarial losses
  mel.py            the log mel the flow produces
  data.py           filelist reader and dataset
  train.py          training loop, checkpoints, exports, previews
  inference.py      export loader, high-band diffuser, render in passes
  rmvpe.py          RMVPE pitch estimator
  slicer.py         silence slicer
```

## Install

Python 3.10+ and PyTorch 2.4 or newer; a CUDA GPU for training.

```
pip install -r requirements.txt
```

## Quick start

```
python preprocess.py path/to/recordings datasets/my_set --sample-rate 44100
python extract.py datasets/my_set
python train.py --name my_vocoder --filelist datasets/my_set/filelist.txt --sample-rate 44100
tensorboard --logdir logs/my_vocoder/eval
python infer.py path/to/song.flac --export logs/my_vocoder/my_vocoder_vocoder_100e_50000s.pth
```

Use the same `--sample-rate` in `preprocess.py` and `train.py`: the trainer
refuses clips at another rate.

## Data

### preprocess.py

Reads every audio file under the folder (wav, flac, mp3, ogg, m4a, opus, aac,
wma) and, per recording:

1. resamples it to `--sample-rate` (44100 by default) in mono;
2. removes the content under 48 Hz;
3. levels it as RVC does (a blend toward a peak of 0.9, so a quiet take stays
   quieter than a loud one);
4. cuts it at silences into clips of up to 3 seconds, overlapping by 0.3 s,
   in `datasets/my_set/wavs/`.

| Option          | Default        | |
|-----------------|----------------|-|
| `--sample-rate` | 44100          | 32000, 40000, 44100 or 48000 |
| `--min-seconds` | 0.5            | shorter clips and recordings are dropped |
| `--format`      | flac           | flac or wav |
| `--workers`     | half the CPUs  | |

### extract.py

Writes the pitch of each clip with RMVPE (100 frames per second, 0 where
unvoiced) to `datasets/my_set/f0/<clip>.npy` and the filelist
`datasets/my_set/filelist.txt`. The RMVPE weights are downloaded to
`assets/rmvpe.pt` the first time. Clips that already have a pitch file are
skipped, so it can be rerun after adding clips.

| Option        | Default                  | |
|---------------|--------------------------|-|
| `--rmvpe`     | `assets/rmvpe.pt`        | RMVPE weights |
| `--device`    | cuda when available      | |
| `--threshold` | 0.03                     | voicing threshold |

### Filelist

One clip per line, paths relative to the filelist's folder (or `--data-root`):

```
wavs/0_0.flac|f0/0_0.npy
wavs/0_1.flac|f0/0_1.npy
```

An RVC experiment's `filelist.txt` (`audio|features|f0|f0_voiced|speaker`)
works as it is: the fourth field is read as the pitch, in Hz at 100 frames per
second. Its audio must still be at the config's rate.

## Train

```
python train.py --name my_vocoder --filelist datasets/my_set/filelist.txt --sample-rate 44100 --epochs 100 --batch-size 8
```

| Option           | Default  | |
|------------------|----------|-|
| `--name`         | required | run name; everything goes to `logs/<name>/` |
| `--filelist`     | required | |
| `--data-root`    | the filelist's folder | where relative paths start |
| `--sample-rate`  | 44100    | rate of a new run; picks `configs/<rate>.json` |
| `--config`       |          | config of a new run, in place of `--sample-rate`'s |
| `--logs`         | `logs`   | |
| `--epochs`       | 100      | |
| `--batch-size`   | 8        | |
| `--precision`    | bf16     | fp32, bf16 or fp16 |
| `--save-every`   | 1        | epochs between checkpoints |
| `--save-steps`   | 0        | also save every this many steps |
| `--pretrained-g` |          | G checkpoint or export to start a new run from |
| `--pretrained-d` |          | D checkpoint to start a new run from |
| `--reference`    |          | clip of the filelist rendered in the previews |
| `--workers`      | 4        | data loader workers |
| `--device`       |          | |

### Stages

The first `aux_epochs` (1, at most a quarter of the run; an older config's
`aux_steps` counts steps instead) train on the mel,
MR-STFT and harmonic losses alone; a run started from `--pretrained-g` skips
them. The point is fixed when the run starts and kept in the checkpoints, so a
resume with more epochs does not move it. The discriminator joins after: compressed multi-resolution
spectrograms, period heads, an envelope head from 3 kHz to Nyquist and
phase heads from 3 kHz up (how each bin's phase advances over a hop and steps
from the bin below), all with SAN heads. Its learning rate starts then, with its own warmup from the base rate,
and decays over the steps it trains.

### Output

Everything goes to `logs/my_vocoder/`:

- `config.json`, copied from `configs/<rate>.json` (or `--config`) on the
  first run. Edit it there to change the model or the losses of that run; the
  rate of a run is fixed once it exists.
- `G_<step>.pth` and `D_<step>.pth`, the latest checkpoints only. Running the
  same command again resumes from them.
- `my_vocoder_vocoder_<epoch>e_<step>s.pth`, the export: EMA weights and the
  mel and model settings (`kind: rectified_vocoder`). This is the file to load
  in ShiroRVC.
- `eval/`, TensorBoard scalars and previews.
- `previews/`, the latest preview as wav files: `generated` with the band
  contrast measured from the clip, `unmeasured` without it (as inference runs
  without a flow that predicts it), and `original`.

### Fine-tuning

To adapt a trained vocoder to a new voice, start a new run from it:

```
python train.py --name my_finetune --filelist datasets/new_voice/filelist.txt \
    --pretrained-g logs/my_vocoder/G_<step>.pth --pretrained-d logs/my_vocoder/D_<step>.pth
```

`--pretrained-g` also takes an export. Both only apply to a new run; a run
with checkpoints resumes from its own. A discriminator family the D file
lacks (one added to the config since) starts fresh, in a resume or a new run.

## Config

The sections of `configs/<rate>.json`:

- `data`: the mel (`hop_length`, `n_fft`, `n_mels`, `mel_fmin`, `mel_fmax`) and
  the normalisation the network sees (`mel_mean`, `mel_std`). It must match the
  flow that will feed the vocoder.
- `model`: the generator. `synthesis_hop × harmonic_substeps` must divide the
  mel hop. `contrast_band_edges` sets the contrast bands; remove it to render
  from the mel alone. `glottal_max_hz` caps the glottal bands.
  `max_phase_spread` (radians, 0 off) lets each band's harmonics take a
  random phase per synthesis step, of a spread the network predicts; an export
  without it loads with the spread starting near 0. `phase_head_hz` (remove it
  to turn it off) lets the network turn the phase of every harmonic bin above
  it, faded in over the 1 kHz below; an export without it loads turning nothing.
  `phase_noise_mix` adds, per bin, how much of a white noise's phase to blend
  into that turn; an export without it loads with the share near 0.
  `control_interpolation` ("linear" or "cubic") is how the gains go from
  frames to synthesis steps; an export without it renders linear.
  `source_phase_hz` (remove it to turn it off) gives every harmonic of the
  harmonic branch from there up a fixed random phase of its own (each bin
  takes that of the harmonic of f0 nearest to it), faded in over the 1 kHz
  below, so the generator trains on scattered harmonics; such an export skips
  the inference diffuser.
- `discriminator`: periods, MRD resolutions, the envelope bands, the phase head
  resolutions (`phase_*`, from `phase_min_hz` up; remove them to turn it off)
  and `use_san`.
- `train`: learning rate and schedule, EMA, `segment_size` (in samples), loss
  weights (`c_mel`, `c_stft`, `c_harmonic`, `c_envelope` with its
  `envelope_loss_bands_hz`), `aux_epochs`, `contrast_dropout`
  (how often the contrast input is dropped, so the model also works without
  it) and the preview and log intervals.

## Inference

```
python infer.py path/to/recordings --export my_vocoder_vocoder_100e_50000s.pth --output outputs
```

Renders each file (or every audio file under a folder) from its own log mel
and RMVPE pitch, as `<name>_chouwagan.wav` at the export's rate: a check of
what the vocoder makes of a mel, the same stage ShiroRVC runs on a flow's
output. Long files render in passes crossfaded at the joins.

The output goes through a high-band diffuser: above 4 kHz (faded in from
3 kHz), each bin of a short STFT turns by a fixed random phase of the
harmonic of f0 nearest to it, so every harmonic turns whole and keeps its
phase as it moves. The pulse train has all its harmonics in phase, which up
there sounds like static; scattered harmonic phases spread each pulse's top
over the cycle and leave the harmonics sharp. It runs only at inference, and
only for an export without `source_phase_hz`, whose source already scatters
the harmonics' phases.

| Option          | Default    | |
|-----------------|------------|-|
| `--export`      | required   | the vocoder export |
| `--output`      | `outputs`  | |
| `--contrast`    | measured   | `measured` from each input, or `none`, as without a flow that predicts it |
| `--no-diffuse`  |            | leave the diffuser out |
| `--rmvpe`       | `assets/rmvpe.pt` | downloaded the first time |
| `--threshold`   | 0.03       | voicing threshold |
| `--device`      | cuda when available | |

From Python:

```python
import torch
from chouwagan.data import f0_to_mel_rate, mel_frames
from chouwagan.inference import load_export, render
from chouwagan.mel import LogMel, normalize_mel

vocoder = load_export("my_vocoder_vocoder_100e_50000s.pth")  # diffuse=False leaves the diffuser out
data = vocoder.data
hop = data["hop_length"]

# audio: [samples] at data["sample_rate"]; f0: [frames] at 100 fps from RMVPE.
frames = min(audio.shape[0] // hop, mel_frames(f0.shape[0], data["sample_rate"], hop))
audio = audio[: frames * hop].unsqueeze(0)
mel = normalize_mel(LogMel.from_config(data)(audio), data)
f0 = f0_to_mel_rate(f0, frames, data["sample_rate"], hop).unsqueeze(0)
wave = render(vocoder, mel, f0, audio=audio)  # [1, frames * hop]; audio= measures the band contrast
```

## License

MIT. Third-party code and its licenses are listed in `THIRD_PARTY_NOTICES`.
