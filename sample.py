# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""i2va inference entry.

Usage::

    python sample.py --config configs/inference/robotwin_i2va.yaml \
                     [--save-root /path/to/output]
"""
import argparse

from src.inference.server import run_i2va
from src.utils import init_logger, logger


def main():
    parser = argparse.ArgumentParser(description="LingBot-VA i2va sampling entry")
    parser.add_argument('--config', required=True, help="Path to YAML i2va config")
    parser.add_argument('--save-root', default=None, help="Override config.save_root")
    args = parser.parse_args()
    run_i2va(args.config, save_root=args.save_root)
    logger.info("sample.py done.")


if __name__ == '__main__':
    init_logger()
    main()
