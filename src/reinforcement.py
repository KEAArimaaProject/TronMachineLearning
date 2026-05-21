#!/usr/bin/env python3
"""
Train a single‑agent reinforcement learning policy for the Tron (Lightcycles) game.

The environment uses the fast vectorized TronBatchModel from 'tron_model.py'.
The agent controls player 0, while opponents (player 1 and optionally more)
follow a fixed policy provided by the reusable controller classes.

Dependencies:
    numpy, gymnasium, stable-baselines3, torch

Usage:
    python train_tron_rl.py --total_timesteps 1_000_000 --model_path tron_ppo.zip
"""

import argparse
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from model import TronBatchModel

# Import the pre‑defined opponent controllers (as used in agent_viewer.py)
# These provide .actions(model) -> (1, players) array.
from controller import RandomController, GreedySpaceController


class OpponentAdapter:
    """
    Adapts a controller (RandomController, GreedySpaceController) to the
    interface required by TronSingleAgentEnv: act(model, env_id) returns
    actions for all opponents (players 1..P-1) in that environment.
    """

    def __init__(self, controller):
        self.controller = controller

    def act(self, model: TronBatchModel, env_id: int = 0) -> np.ndarray:
        """
        Returns actions for opponents in the given environment.
        """
        # controller.actions(model) returns shape (1, players) – for env 0 only.
        all_actions = self.controller.actions(model)  # (1, players)
        # Discard the agent's own action (player 0) – keep only opponents.
        return all_actions[env_id, 1:].copy()


class TronSingleAgentEnv(gym.Env):
    """
    Gymnasium environment for a single RL agent controlling player 0 in Tron.

    Opponents are controlled by any object that provides .actions(model)
    (e.g. RandomController, GreedySpaceController) or by a string policy name.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        width: int = 48,
        height: int = 32,
        players: int = 2,
        max_steps: Optional[int] = None,
        opponent_policy: str = "random",          # "random" or "greedy"
        # Alternatively, pass a pre‑instantiated controller
        opponent_controller=None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.players = players
        self.max_steps = max_steps or (width * height)
        self.render_mode = render_mode
        self.seed = seed

        # Internal vectorized model (single environment)
        self.board_model = TronBatchModel(
            width=width,
            height=height,
            players=players,
            envs=1,
            max_steps=self.max_steps,
            keep_owner=True,
            randomize_spawns=True,
            seed=seed,
        )

        # Set up opponent controller
        if opponent_controller is not None:
            self.opponent = OpponentAdapter(opponent_controller)
        else:
            if opponent_policy == "random":
                raw_controller = RandomController(seed=seed)
            elif opponent_policy == "greedy":
                raw_controller = GreedySpaceController()
            else:
                raise ValueError(f"Unknown opponent policy: {opponent_policy}")
            self.opponent = OpponentAdapter(raw_controller)

        # Observation space from `observe_lite` for player 0
        self.board_model.reset()
        obs_lite = self.board_model.observe_lite()[0, 0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_lite.shape, dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)
        self._render_image = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.board_model.rng = np.random.default_rng(seed)

        self.board_model.reset(env_ids=[0])
        obs = self.board_model.observe_lite()[0, 0]
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Build action array for all players
        actions = np.zeros((1, self.players), dtype=np.int8)
        actions[0, 0] = action
        opponents_actions = self.opponent.act(self.board_model, env_id=0)
        actions[0, 1:] = opponents_actions

        result = self.board_model.step(actions)

        reward = float(result.reward[0, 0])
        done_game = result.done[0]
        agent_alive = result.alive[0, 0]
        terminated = done_game or not agent_alive

        info = {
            "done": bool(result.done[0]),
            "alive": result.alive[0].copy(),
            "winner": None,
        }
        if done_game:
            alive_players = np.where(result.alive[0])[0]
            info["winner"] = int(alive_players[0]) if len(alive_players) == 1 else -1

        obs = self.board_model.observe_lite()[0, 0]
        return obs, reward, terminated, False, info

    def render(self) -> Optional[np.ndarray]:
        """Render the board (same as original)."""
        if self.render_mode is None:
            return None

        owner = self.board_model.owner[0] if self.board_model.owner is not None else None
        if owner is None:
            owner = np.zeros((self.height, self.width), dtype=np.uint8)
            for p in range(self.players):
                if self.board_model.alive[0, p]:
                    px, py = self.board_model.pos[0, p]
                    if 0 <= px < self.width and 0 <= py < self.height:
                        owner[py, px] = p + 1
        else:
            owner = owner.copy()

        colours = np.array([
            [0, 0, 0],
            [0, 255, 255],
            [255, 0, 255],
            [0, 255, 0],
            [255, 255, 0],
        ], dtype=np.uint8)

        img = colours[owner]
        for p in range(self.players):
            if self.board_model.alive[0, p]:
                px, py = self.board_model.pos[0, p]
                if 0 <= px < self.width and 0 <= py < self.height:
                    img[py, px] = [255, 255, 255]

        if self.render_mode == "rgb_array":
            return img
        elif self.render_mode == "human":
            try:
                import matplotlib.pyplot as plt
                if not hasattr(self, "_fig"):
                    plt.ion()
                    self._fig, self._ax = plt.subplots(figsize=(self.width/4, self.height/4))
                    self._ax.set_title("Tron Game")
                    self._img_handle = self._ax.imshow(img, interpolation="nearest")
                    plt.show()
                else:
                    self._img_handle.set_data(img)
                    self._fig.canvas.flush_events()
            except ImportError:
                print("matplotlib not installed; cannot render to human window.")
        return None

    def close(self):
        if hasattr(self, "_fig"):
            plt.close(self._fig)


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent for Tron.")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--model_path", type=str, default="tron_ppo.zip")
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "greedy"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="Render after training.")
    args = parser.parse_args()

    env = TronSingleAgentEnv(
        width=args.width,
        height=args.height,
        players=args.players,
        opponent_policy=args.opponent,
        render_mode="human" if args.render else None,
        seed=args.seed,
    )

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./tron_tensorboard/",
        seed=args.seed,
    )

    print(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

    if args.render:
        print("Evaluating trained agent (press Ctrl+C to stop)...")
        obs = vec_env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = vec_env.step(action)
            env.render()
            if dones[0]:
                obs = vec_env.reset()


if __name__ == "__main__":
    main()