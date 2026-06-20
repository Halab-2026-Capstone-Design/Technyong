"""Serve preset E: rolling inpainting plus temporal ensemble."""

import logging

import tyro

from serve_b1k import Args, apply_preset, main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(apply_preset(tyro.cli(Args), "E"))
