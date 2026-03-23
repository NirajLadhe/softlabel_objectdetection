"""
CLI entry point: softlabel-train

Delegates to the vanilla PyTorch training orchestrator.
For Lightning training, use: softlabel-train-lightning
"""

import argparse
import os

from softlabel_od.object_detection.utils.config import get_cfg_defaults, get_cfg_from_file
from softlabel_od.object_detection.train import main as train_main

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "1d_softlabel_config_albers.yaml")


def main():
    parser = argparse.ArgumentParser(description="Train VinDr-CXR model (vanilla PyTorch)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to config YAML")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        print(f"Available configs are in: {os.path.join(PROJECT_ROOT, 'configs/')}")
        return

    cfg = get_cfg_from_file(args.config)
    cfg.defrost()
    cfg.CONFIG_FILE_PATH = args.config
    cfg.freeze()
    train_main(cfg)


if __name__ == "__main__":
    main()
