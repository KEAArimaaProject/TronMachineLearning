"""
Optimized Tron / Lightcycles batch simulation.

Improvements over the original:
- O(P²) -> O(P) head‑to‑head collision detection via position encoding + hashing.
- Reused legal_actions buffer to reduce allocations.
- Precomputed normalization denominators.
- Optional Numba acceleration for collision detection.
- Better memory layout (C‑order) and __slots__.
- Fixed deprecated np.bool_ usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

# Direction encoding: 0 up, 1 right, 2 down, 3 left
DIR_VECTORS = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.int16)
TURN = np.array([0, -1, 1], dtype=np.int8)          # action -> heading delta

# Try to import Numba for even faster collision detection
try:
    from numba import jit

    @jit(nopython=True, cache=True)
    def _detect_head_collisions_numba(x, y, valid, width, height):
        """
        Numba-accelerated head-to-head collision detection.
        Array reused across environments to avoid allocations.
        """
        envs, players = x.shape
        head_hit = np.zeros((envs, players), dtype=np.bool_)

        # Pre‑allocate once, outside the environment loop
        max_positions = width * height
        pos_to_player = np.full(max_positions, -1, dtype=np.int32)

        for e in range(envs):
            # Reset the mapping array for this environment
            pos_to_player[:] = -1

            for p in range(players):
                if valid[e, p]:
                    pos_key = int(y[e, p]) * width + int(x[e, p])  # Correct hash
                    prev_player = pos_to_player[pos_key]
                    if prev_player != -1:
                        head_hit[e, p] = True
                        head_hit[e, prev_player] = True
                    else:
                        pos_to_player[pos_key] = p

        return head_hit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy function – never used when Numba is missing (use_numba is forced to False)
    def _detect_head_collisions_numba(*args, **kwargs):
        return None


@dataclass(slots=True)
class StepResult:
    reward: np.ndarray          # float32 [envs, players]
    done: np.ndarray            # bool [envs]
    alive: np.ndarray           # bool [envs, players]
    died: np.ndarray            # bool [envs, players]


@dataclass(slots=True)
class Replay:
    """A small view-only recording of one completed game."""

    width: int
    height: int
    players: int
    owner_frames: list[np.ndarray]
    head_frames: list[np.ndarray]
    alive_frames: list[np.ndarray]

    @classmethod
    def empty(cls, model: "TronBatchModel") -> "Replay":
        return cls(model.width, model.height, model.players, [], [], [])

    def append(self, model: "TronBatchModel", env: int = 0) -> None:
        if model.owner is None:
            owner = model.occupied[env].astype(np.uint8) * 5
        else:
            owner = model.owner[env].copy()
        self.owner_frames.append(owner)
        self.head_frames.append(model.pos[env].copy())
        self.alive_frames.append(model.alive[env].copy())

    def __len__(self) -> int:
        return len(self.owner_frames)


class TronBatchModel:
    """
    Vectorized Lightcycles simulation with optimised collision detection.
    """

    __slots__ = (
        "width", "height", "players", "envs", "max_steps", "keep_owner",
        "randomize_spawns", "rng", "_width_norm", "_height_norm",
        "occupied", "owner", "pos", "heading", "alive", "done", "tick",
        "_legal_cache", "_use_numba"
    )

    def __init__(
        self,
        width: int = 48,
        height: int = 32,
        players: int = 2,
        envs: int = 1,
        max_steps: Optional[int] = None,
        keep_owner: bool = False,
        randomize_spawns: bool = True,
        seed: Optional[int] = None,
        use_numba: bool = True,
    ) -> None:
        if not (2 <= players <= 4):
            raise ValueError("players must be 2, 3, or 4")
        if width < 8 or height < 8:
            raise ValueError("width and height must be at least 8")

        self.width = int(width)
        self.height = int(height)
        self.players = int(players)
        self.envs = int(envs)
        self.max_steps = int(max_steps or (width * height))
        self.keep_owner = bool(keep_owner)
        self.randomize_spawns = bool(randomize_spawns)
        self.rng = np.random.default_rng(seed)
        self._use_numba = use_numba and NUMBA_AVAILABLE

        # Precomputed normalization denominators
        self._width_norm = max(1, width - 1)
        self._height_norm = max(1, height - 1)

        # Core buffers: C‑order for better cache locality
        self.occupied = np.zeros((envs, height, width), dtype=bool, order='C')
        self.owner = np.zeros((envs, height, width), dtype=np.uint8, order='C') if keep_owner else None
        self.pos = np.zeros((envs, players, 2), dtype=np.int16, order='C')
        self.heading = np.zeros((envs, players), dtype=np.int8, order='C')
        self.alive = np.ones((envs, players), dtype=bool, order='C')
        self.done = np.zeros(envs, dtype=bool, order='C')
        self.tick = np.zeros(envs, dtype=np.int32, order='C')

        # Reusable buffer for legal_actions()

        self.reset()

    def reset(self, env_ids: Optional[Iterable[int] | np.ndarray] = None) -> None:
        """Reset all envs, or only env_ids."""
        if env_ids is None:
            ids = np.arange(self.envs)
        else:
            ids = np.asarray(list(env_ids), dtype=np.int64)

        self.occupied[ids] = False
        if self.owner is not None:
            self.owner[ids] = 0
        self.alive[ids] = True
        self.done[ids] = False
        self.tick[ids] = 0

        base_pos, base_heading = self._spawn_layout()
        pos = np.broadcast_to(base_pos[None, :, :], (len(ids), self.players, 2)).copy()

        # Small random jitter improves training diversity while avoiding walls.
        if self.randomize_spawns:
            jitter_x = max(1, self.width // 16)
            jitter_y = max(1, self.height // 16)
            jitter = self.rng.integers(
                low=[-jitter_x, -jitter_y],
                high=[jitter_x + 1, jitter_y + 1],
                size=(len(ids), self.players, 2),
                dtype=np.int16,
            )
            pos += jitter
            pos[..., 0] = np.clip(pos[..., 0], 2, self.width - 3)
            pos[..., 1] = np.clip(pos[..., 1], 2, self.height - 3)

        self.pos[ids] = pos
        self.heading[ids] = base_heading[None, :]

        # Mark starting cells occupied.
        e = ids[:, None]
        p = np.arange(self.players)[None, :]
        x = self.pos[ids, :, 0]
        y = self.pos[ids, :, 1]
        self.occupied[e, y, x] = True
        if self.owner is not None:
            self.owner[e, y, x] = p + 1

    def _spawn_layout(self) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic spawn positions/headings aiming toward the center."""
        w, h, p = self.width, self.height, self.players
        layouts = {
            2: ([(w // 4, h // 2), (3 * w // 4, h // 2)], [1, 3]),
            3: ([(w // 4, h // 2), (3 * w // 4, h // 2), (w // 2, h // 4)], [1, 3, 2]),
            4: (
                [(w // 4, h // 4), (3 * w // 4, 3 * h // 4),
                 (3 * w // 4, h // 4), (w // 4, 3 * h // 4)],
                [1, 3, 2, 0],
            ),
        }
        pos, heading = layouts[p]
        return np.asarray(pos, dtype=np.int16), np.asarray(heading, dtype=np.int8)

    def step(self, actions: np.ndarray) -> StepResult:
        """
        Advance every environment one tick.

        Args:
            actions: int array [envs, players] or [players].
                     0 straight, 1 left, 2 right.
        """
        actions = np.asarray(actions, dtype=np.int8)
        if actions.ndim == 1:
            actions = np.broadcast_to(actions[None, :], (self.envs, self.players))
        if actions.shape != (self.envs, self.players):
            raise ValueError(f"actions must have shape {(self.envs, self.players)}")
        actions = np.clip(actions, 0, 2)

        reward = np.zeros((self.envs, self.players), dtype=np.float32)
        active = (~self.done)[:, None] & self.alive
        if not active.any():
            died = np.zeros((self.envs, self.players), dtype=bool)
            return StepResult(reward, self.done.copy(), self.alive.copy(), died)

        # Move
        new_heading = (self.heading + TURN[actions]) & 3
        delta = DIR_VECTORS[new_heading]
        new_pos = self.pos + delta
        x = new_pos[..., 0]
        y = new_pos[..., 1]

        in_bounds = (0 <= x) & (x < self.width) & (0 <= y) & (y < self.height)

        # Trail hits
        trail_hit = np.zeros((self.envs, self.players), dtype=bool)
        e_idx = np.broadcast_to(np.arange(self.envs)[:, None], (self.envs, self.players))
        valid = active & in_bounds
        trail_hit[valid] = self.occupied[e_idx[valid], y[valid], x[valid]]

        # Head‑to‑head collisions – OPTIMISED O(P) version (hashing bug FIXED)
        if self._use_numba:
            head_hit = _detect_head_collisions_numba(x, y, valid, self.width, self.height)
        else:
            head_hit = self._detect_head_collisions_python(x, y, valid)

        died = active & ((~in_bounds) | trail_hit | head_hit)
        survived = active & ~died

        reward[died] = -1.0

        # Apply movement
        self.heading[active] = new_heading[active]
        self.pos[survived] = new_pos[survived]

        # Mark new cells
        self.occupied[e_idx[survived], y[survived], x[survived]] = True
        if self.owner is not None:
            p_idx = np.broadcast_to(np.arange(self.players)[None, :], (self.envs, self.players))
            self.owner[e_idx[survived], y[survived], x[survived]] = p_idx[survived] + 1

        self.alive[died] = False
        self.tick[~self.done] += 1

        alive_count = self.alive.sum(axis=1)
        newly_done = (~self.done) & ((alive_count <= 1) | (self.tick >= self.max_steps))

        # Winner reward only on terminal combat states, not on max-step draws.
        has_winner = newly_done & (alive_count == 1)
        if has_winner.any():
            reward[has_winner] += self.alive[has_winner].astype(np.float32)

        self.done[newly_done] = True
        return StepResult(reward, self.done.copy(), self.alive.copy(), died)

    def _detect_head_collisions_python(self, x, y, valid):
        """Pure Python fallback (O(P) per env) – CORRECT hash."""
        print("Fallback collision used")
        envs, players = x.shape
        head_hit = np.zeros((envs, players), dtype=bool)

        for e in range(envs):
            pos_map = {}
            for p in range(players):
                if valid[e, p]:
                    key = int(y[e, p]) * self.width + int(x[e, p])   # y * width + x
                    if key in pos_map:
                        head_hit[e, p] = True
                        head_hit[e, pos_map[key]] = True
                    else:
                        pos_map[key] = p
        return head_hit


    def observe_lite(self) -> np.ndarray:
        """
        Compact float observation [envs, players, features].

        Changes:
        - The 3-way legal-action booleans are replaced by distances (normalized)
          to the nearest wall along straight/left/right directions.
        - For each other player we include: rel_x, rel_y (normalized), rel_alive,
          and the opponent heading one-hot (4 dims).

        Features per player:
          distances(3)  -- distance for straight/left/right
          x_norm, y_norm (2)
          heading one-hot (4)
          alive (1)
          for each other player:
            rel_x, rel_y (2), rel_alive (1), rel_heading_onehot (4) => 7*(players-1)
        """
        # --- Distance-to-wall for each candidate direction (ray marching) ---
        candidate_heading = (self.heading[:, :, None] + TURN[None, None, :]) & 3
        candidate_delta = DIR_VECTORS[candidate_heading]  # int deltas

        envs, players = self.envs, self.players
        max_search = max(self.width, self.height)

        pos0 = self.pos[:, :, None, :].astype(np.int32)  # (envs, players, 1, 2)
        delta = candidate_delta.astype(np.int32)  # (envs, players, 3, 2)

        ray_valid = (self.alive[:, :, None] & (~self.done[:, None, None]))  # (envs, players, 1)
        ray_valid = np.broadcast_to(ray_valid, (envs, players, 3))

        distances = np.full((envs, players, 3), max_search, dtype=np.int32)
        still_searching = ray_valid.copy()

        # Ray march steps 1..max_search
        for s in range(1, max_search + 1):
            pos_s = pos0 + delta * s  # (envs, players, 3, 2)
            x_s = pos_s[..., 0]
            y_s = pos_s[..., 1]

            in_bounds_s = (0 <= x_s) & (x_s < self.width) & (0 <= y_s) & (y_s < self.height)
            hit_s = ~in_bounds_s.copy()  # out-of-bounds counts as a hit

            # Check occupied only where in-bounds and still searching
            mask = in_bounds_s & still_searching
            if mask.any():
                idx = np.where(mask)
                xs = x_s[idx]
                ys = y_s[idx]
                env_idx = idx[0]
                occ_vals = self.occupied[env_idx, ys, xs]
                hit_s[idx] = occ_vals

            new_hits = still_searching & hit_s
            if new_hits.any():
                distances[new_hits] = s
                still_searching[new_hits] = False

            if not still_searching.any():
                break

        distances[~ray_valid] = 0
        distances_norm = 1 / (distances.astype(np.float32) + 1)

        # --- Position / heading / alive parts ---
        xy = self.pos.astype(np.float32)
        xy[..., 0] /= self._width_norm
        xy[..., 1] /= self._height_norm
        heading_oh = np.eye(4, dtype=np.float32)[self.heading]  # (envs, players, 4)
        alive = self.alive[..., None].astype(np.float32)

        # --- Relative features to other players: rel_x, rel_y, rel_alive, rel_heading_onehot ---
        heading_oh_all = heading_oh  # alias for clarity
        rel_parts = []
        for i in range(self.players):
            parts = []
            for j in range(self.players):
                if i == j:
                    continue
                # relative dxy normalized
                dxy = (self.pos[:, j] - self.pos[:, i]).astype(np.float32)
                dxy[:, 0] /= self._width_norm
                dxy[:, 1] /= self._height_norm

                alive_j = self.alive[:, j:j + 1].astype(np.float32)  # (envs,1)
                heading_j = heading_oh_all[:, j, :]  # (envs,4)

                # concat: dxy (2), alive_j (1), heading_j (4) -> 7 cols
                parts.append(np.concatenate([dxy, alive_j, heading_j], axis=1))
            rel_parts.append(np.concatenate(parts, axis=1))
        rel = np.stack(rel_parts, axis=1)  # (envs, players, 7*(players-1))

        # final observation concat: distances_norm (3), xy (2), heading_oh (4), alive(1), rel
        return np.concatenate([distances_norm, xy, heading_oh, alive, rel], axis=2)


    def auto_reset_done(self) -> np.ndarray:
        """Reset terminal envs and return their ids."""
        ids = np.flatnonzero(self.done)
        if len(ids):
            self.reset(ids)
        return ids