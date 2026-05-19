
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from model import TronBatchModel
from controller import Controller, GreedySpaceController
from src.view import GameView


class LLMController:
    """ LEGACY :3
    Optional LLM-facing controller for a single rendered game.

    This is deliberately NOT for high-throughput training. It converts one env into
    a compact text state, calls a user-supplied llm_fn(prompt) -> str, and parses a
    simple action. Use it for experiments, explanations, or human-vs-agent demos.

    Expected LLM output can be any string containing one of:
        "straight", "left", "right", "0", "1", or "2".
    """

    def __init__(self, llm_fn: Callable[[str], str], player: int = 0, fallback: Controller | None = None) -> None:
        self.llm_fn = llm_fn
        self.player = int(player)
        self.fallback = fallback or GreedySpaceController()

    def actions(self, model: Any) -> np.ndarray:
        acts = self.fallback.actions(model)
        if model.envs != 1:
            raise ValueError("LLMController is intended for envs=1 only")
        prompt = self.make_prompt(model, self.player)
        acts[0, self.player] = self.parse_action(self.llm_fn(prompt))
        return acts

    @staticmethod
    def parse_action(text: str) -> np.int8:
        t = text.strip().lower()
        if "left" in t or t == "1":
            return np.int8(1)
        if "right" in t or t == "2":
            return np.int8(2)
        return np.int8(0)

    @staticmethod
    def make_prompt(model: Any, player: int = 0) -> str:
        legal = model.legal_actions()[0, player]
        x, y = model.pos[0, player]
        heading = int(model.heading[0, player])
        heading_name = ["up", "right", "down", "left"][heading]
        alive = bool(model.alive[0, player])

        others = []
        for p in range(model.players):
            if p == player:
                continue
            ox, oy = model.pos[0, p]
            others.append(f"P{p + 1}=({int(ox)},{int(oy)}), alive={bool(model.alive[0, p])}")

        legal_names = [name for ok, name in zip(legal, ["straight", "left", "right"], strict=False) if ok]
        return (
            "You control a Tron lightcycle. Choose exactly one action: straight, left, or right.\n"
            f"Board: {model.width}x{model.height}.\n"
            f"You are P{player + 1} at ({int(x)},{int(y)}), heading {heading_name}, alive={alive}.\n"
            f"Legal one-step actions: {', '.join(legal_names) or 'none'}.\n"
            f"Other players: {'; '.join(others)}.\n"
            "Reply with only the action."
        )


from agent_classes import TronAgent

from typing import List, Optional
from dataclasses import dataclass
import torch

class PyTorchTronAgent(TronAgent):
    """
    Wraps a PyTorch model for use in TronBatchModel.

    Parameters
    ----------
    model : torch.nn.Module
        A callable that takes a batch of observations and returns Q‑values or action probabilities.
        - For Q‑values: shape [batch, 3]  → greedy action.
        - For probabilities: shape [batch, 3] → sample or argmax.
    obs_type : 'lite' | 'grid'
        Must match the observation the model was trained on.
    device : str or torch.device
    deterministic : bool
        If True, always argmax; if False, sample from softmax (for PPO‑style).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        obs_type: str = "lite",
        device: str = "cpu",
        deterministic: bool = True,
    ):
        self.model = model.to(device).eval()
        self._obs_type = obs_type
        self.device = device
        self.deterministic = deterministic

    @property
    def observation_type(self) -> str:
        return self._obs_type

    @torch.no_grad()
    def act(self, observation: np.ndarray, legal_actions: np.ndarray) -> np.ndarray:
        # Convert to tensor
        if observation.dtype == np.float64:       # ensure float32
            observation = observation.astype(np.float32)
        obs_t = torch.as_tensor(observation, device=self.device)

        # Pass through model
        output = self.model(obs_t)                # [batch, 3] expected

        # Mask illegal actions
        legal_t = torch.as_tensor(legal_actions, device=self.device)
        output = output.masked_fill(~legal_t, float("-inf"))

        if self.deterministic:
            actions = output.argmax(dim=-1)       # greedy
        else:
            probs = torch.softmax(output, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)

        return actions.cpu().numpy().astype(np.int8)


@dataclass
class BattleResult:
    wins: List[int]  # count per agent
    draws: int
    steps: int

class TronBattleRunner:
    """
    Runs head‑to‑head or free‑for‑all battles between agents.
    """

    def __init__(
            self,
            width: int = 32,
            height: int = 32,
            max_steps: int = 500,
            envs: int = 1024,  # parallel envs for speed
            seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.envs = envs
        self.seed = seed

    def battle(
            self,
            agents: List[TronAgent],
            episodes: int = 100,
            render: bool = False,
    ) -> BattleResult:
        """
        Pit the given agents against each other.

        agents length must be 2, 3, or 4. The runner assigns agent[i] to player i.
        All envs play the same matchup simultaneously.
        """
        players = len(agents)
        if players < 2 or players > 4:
            raise ValueError("Need 2‑4 agents")

        model = TronBatchModel(
            width=self.width,
            height=self.height,
            players=players,
            envs=self.envs,
            max_steps=self.max_steps,
            keep_owner=render,
            seed=self.seed,
        )

        wins = [0] * players
        draws = 0
        total_steps = 0

        remaining_episodes = episodes
        while remaining_episodes > 0:
            model.reset()
            batch_size = min(self.envs, remaining_episodes)
            # we can run multiple episodes per loop by using only the first batch_size envs
            env_mask = np.arange(batch_size)

            # Wrap agents for the batch if needed (see note below)
            while not np.any(model.done[env_mask]):
                actions = np.zeros((batch_size, players), dtype=np.int8)
                for p, agent in enumerate(agents):
                    # Get observation for this player (all envs)
                    obs = None
                    try:
                        observation_type = agent.observation_type

                    except AttributeError:
                        observation_type = agent.observation_type
                    if observation_type == "lite":
                        obs = model.observe_lite()  # [envs, players, feat]
                        obs = obs[env_mask, p]  # shape [batch, feat]
                    else:  # grid
                        obs = model.observe_grid()  # [envs, 1+players, h, w]
                        obs = obs[env_mask]  # shape [batch, 1+players, h, w]
                    # Legal actions for this player
                    legal = model.legal_actions()[env_mask, p]  # [batch, 3]

                    actions[:, p] = agent.act(obs, legal)

                # Only step the envs we actually use
                full_actions = np.zeros((self.envs, players), dtype=np.int8)
                full_actions[env_mask] = actions
                result = model.step(full_actions)

            # Record outcomes for the batch
            for e in range(batch_size):
                if np.sum(model.alive[e]) == 1:
                    winner = np.argmax(model.alive[e])
                    wins[winner] += 1
                else:
                    draws += 1  # no single survivor → draw
                total_steps += model.tick[e]

            remaining_episodes -= batch_size

        return BattleResult(wins=wins, draws=draws, steps=total_steps // episodes)
