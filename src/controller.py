"""
Controller layer for the Tron environment with no pygame dependency.

Action encoding:
    0 = straight, 1 = turn left, 2 = turn right
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol

import numpy as np
import time

from model import DIR_VECTORS, TURN


class Controller(Protocol):
    def actions(self, model: Any) -> np.ndarray:
        """Return int actions [envs, players]."""


class RandomController:
    """Fast legal-ish random baseline for large batch rollouts."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def actions(self, model: Any) -> np.ndarray:
        legal = model.legal_actions()
        acts = self.rng.integers(0, 3, size=(model.envs, model.players), dtype=np.int8)

        chosen_is_legal = np.take_along_axis(legal, acts[..., None], axis=2)[..., 0]
        fallback = legal.argmax(axis=2).astype(np.int8)
        any_legal = legal.any(axis=2)
        return np.where((~chosen_is_legal) & any_legal, fallback, acts).astype(np.int8)


class GreedySpaceController:
    """
    Tiny baseline policy: choose the action with longest ray distance before impact.
    Useful for demos, debugging, and weak scripted opponents.
    """

    def observation_type(self) -> str:
        return "lite"

    def actions(self, model: Any) -> np.ndarray:
        scores = np.stack([self._ray_score(model, a) for a in range(3)], axis=2)
        return scores.argmax(axis=2).astype(np.int8)

    @staticmethod
    def _ray_score(model: Any, action: int) -> np.ndarray:
        heading = (model.heading + TURN[action]) & 3
        d = DIR_VECTORS[heading]
        x = model.pos[..., 0].astype(np.int32)
        y = model.pos[..., 1].astype(np.int32)
        score = np.zeros((model.envs, model.players), dtype=np.int16)
        active = model.alive & (~model.done[:, None])

        for _ in range(max(model.width, model.height)):
            x = x + d[..., 0]
            y = y + d[..., 1]
            ok = active & (0 <= x) & (x < model.width) & (0 <= y) & (y < model.height)
            hit = np.zeros_like(ok)
            e = np.broadcast_to(np.arange(model.envs)[:, None], (model.envs, model.players))
            hit[ok] = model.occupied[e[ok], y[ok], x[ok]]
            step_ok = ok & ~hit
            score += step_ok.astype(np.int16)
            active &= step_ok
            if not active.any():
                break
        return score

class FastGreedyController:
    """Chooses action with longest ray distance using precomputed environment features."""
    def actions(self, model):
        # observe_lite returns (envs, players, 3) distances as first 3 features
        dist = model.observe_lite()[..., :3]   # shape (envs, players, 3)
        return dist.argmax(axis=-1).astype(np.int8)

class KeyStateController:
    """
    Human controller driven by the view's pressed-key set.

    Player 1: A left, D right
    Player 2: LEFT left, RIGHT right
    Player 3: J left, L right
    Player 4: F left, H right
    """

    def __init__(
        self,
        pressed: set[str],
        fallback: Optional[Controller] = None,
        human_players: Optional[Iterable[int]] = None,
        delay_seconds: Optional[float] = None,
    ) -> None:
        self.pressed = pressed
        self.fallback = fallback or GreedySpaceController()
        self.human_players = set(human_players) if human_players is not None else {0}
        self.delay_seconds = delay_seconds or 0.1
        self.last_turn_time = [0.0] * 4  # for up to 4 players

    def actions(self, model: Any) -> np.ndarray:
        acts = self.fallback.actions(model)
        keymap = [
            ("a", "d"),
            ("left_arrow", "right_arrow"),
            ("j", "l"),
            ("f", "h"),
        ]
        for p, (left_key, right_key) in enumerate(keymap[:model.players]):
            if p not in self.human_players:
                continue
            current_time = time.time()
            if left_key in self.pressed and current_time - self.last_turn_time[p] > self.delay_seconds:
                acts[0, p] = 1
                self.last_turn_time[p] = current_time
            elif right_key in self.pressed and current_time - self.last_turn_time[p] > self.delay_seconds:
                acts[0, p] = 2
                self.last_turn_time[p] = current_time
            else:
                acts[0, p] = 0
        return acts
