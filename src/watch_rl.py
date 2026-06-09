#!/usr/bin/env python3
import argparse
import numpy as np
from stable_baselines3 import PPO
from trainer import watch_mixed_policy
from controller import GreedySpaceController

class RLWatchWrapper:
    def __init__(self, model_path, player_id=0, device="auto"):
        self.model = PPO.load(model_path, device=device)
        self.player_id = player_id

    def actions(self, model_env):
        envs, players = model_env.envs, model_env.players
        obs = model_env.observe_lite()
        acts = np.zeros((envs, players), dtype=np.int8)
        for e in range(envs):
            action, _ = self.model.predict(obs[e, self.player_id], deterministic=True)
            acts[e, self.player_id] = action
        return acts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to saved PPO .zip file")
    parser.add_argument("--opponent", choices=["self", "greedy"], default="greedy")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--scale", type=int, default=16)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    rl_agent = RLWatchWrapper(args.model_path, player_id=0)

    if args.opponent == "greedy":
        watch_mixed_policy([rl_agent, GreedySpaceController()], players=2,
                           width=args.width, height=args.height,
                           scale=args.scale, fps=args.fps, seed=0)
    else:
        # Self-play: RL controls both players (same model)
        watch_mixed_policy([rl_agent, rl_agent], players=2,
                           width=args.width, height=args.height,
                           scale=args.scale, fps=args.fps, seed=0)

if __name__ == "__main__":
    main()