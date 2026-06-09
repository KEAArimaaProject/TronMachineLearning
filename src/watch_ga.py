#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from model import TronBatchModel
from trainer import MLPPolicy, watch_policy, watch_mixed_policy
from controller import GreedySpaceController

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("genome_path", help="Path to .npy genome file")
    parser.add_argument("--hidden", type=int, required=True, help="Hidden layer size used in training")
    parser.add_argument("--opponent", choices=["self", "greedy"], default="self")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--scale", type=int, default=16)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    probe = TronBatchModel(width=args.width, height=args.height, players=2, envs=1)
    obs_dim = probe.observe_lite().shape[-1]
    genome = np.load(args.genome_path)
    policy = MLPPolicy(obs_dim, hidden=args.hidden, genome=genome)

    if args.opponent == "greedy":
        watch_mixed_policy([policy, GreedySpaceController()], players=2,
                           width=args.width, height=args.height,
                           scale=args.scale, fps=args.fps, seed=0)
    else:
        watch_policy(policy, players=2, width=args.width, height=args.height,
                     scale=args.scale, fps=args.fps, seed=0)

if __name__ == "__main__":
    main()