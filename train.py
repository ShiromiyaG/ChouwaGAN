"""Train ChouwaGAN on a filelist made by extract.py (or an RVC filelist)."""

import argparse

from chouwagan.train import SAMPLE_RATES, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Run name; everything goes to logs/<name>/.")
    parser.add_argument("--filelist", required=True)
    parser.add_argument("--data-root", default=None, help="Where relative paths start; the filelist's folder by default.")
    parser.add_argument("--sample-rate", type=int, default=None, choices=SAMPLE_RATES,
                        help="Rate of a new run: its config is configs/<rate>.json. 44100 by default.")
    parser.add_argument("--config", default=None, help="Config for a new run, in place of --sample-rate's.")
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", default="bf16", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--save-every", type=int, default=1, help="Epochs between checkpoints.")
    parser.add_argument("--save-steps", type=int, default=0, help="Also save every this many steps.")
    parser.add_argument("--pretrained-g", default="", help="Generator to start a new run from: a G checkpoint or an export.")
    parser.add_argument("--pretrained-d", default="", help="Discriminator to start a new run from: a D checkpoint.")
    parser.add_argument("--reference", default="", help="Clip of the filelist rendered in the previews.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
