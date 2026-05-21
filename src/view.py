"""
View layer for rendering Tron games with tkinter.

No third-party rendering dependency is required. The model does not import this file.

About GameView.poll():
    Tkinter needs its event queue processed repeatedly. poll() does that work and
    returns False when the user closes the window. The app loop is therefore:

        while view.poll():
            ...
            view.render(model)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from model import Replay


class GameView:
    """Simple tkinter renderer for live games and completed replays."""

    COLORS = [
        "#1e1e23",  # background
        "#46d2ff",  # player 1
        "#ff5f5f",  # player 2
        "#ffdc50",  # player 3
        "#8cff8c",  # player 4
        "#50505a",  # unknown trail
    ]

    def __init__(self, width_or_model: int | Any, height: int | None = None, scale: int = 16, fps: int = 30) -> None:
        import tkinter as tk

        if height is None:
            self.width = int(width_or_model.width)
            self.height = int(width_or_model.height)
        else:
            self.width = int(width_or_model)
            self.height = int(height)

        self.tk = tk
        self.scale = int(scale)
        self.fps = int(fps)
        self.closed = False
        self.restart_requested = False
        self.pressed: set[str] = set()
        self.just_pressed: set[str] = set()

        # Observation display settings
        self.show_obs = True          # press 'o' to toggle
        self.obs_player = 0           # press 'p' to cycle
        self.obs_label = None         # will be created below

        self.root = tk.Tk()
        self.root.title("MVC Tron")

        self.canvas = tk.Canvas(
            self.root,
            width=self.width * self.scale,
            height=self.height * self.scale,
            bg=self.COLORS[0],
            highlightthickness=0,
        )
        self.canvas.pack()

        self.status = tk.StringVar(value="")
        self.status_label = tk.Label(self.root, textvariable=self.status, font=("Arial", 14, "bold"))
        self.status_label.pack(fill="x", pady=(6, 2))

        self.help_label = tk.Label(
            self.root,
            text="Controls: P1 A/D, P2 arrows, P3 J/L, P4 F/H. Restart: button or R. O: toggle obs, P: cycle player.",
            font=("Arial", 10),
        )
        self.help_label.pack(fill="x")

        self.restart_button = tk.Button(self.root, text="Restart game", command=self.request_restart)
        self.restart_button.pack(fill="x", padx=8, pady=6)

        # Observation display panel (initially hidden, will be shown when toggled)
        self.obs_frame = tk.LabelFrame(self.root, text="Observe Lite", font=("Arial", 10, "bold"))
        self.obs_text = tk.Text(self.obs_frame, height=8, width=70, font=("Courier", 8), wrap="word")
        self.obs_text.pack(padx=4, pady=4, fill="both", expand=True)
        # Initially pack it but hide by toggling later; easier to pack now and later pack_forget
        self.obs_frame.pack(fill="x", padx=8, pady=4)
        self.obs_frame.pack_forget()   # hidden until 'o' is pressed

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.focus_set()

    def request_restart(self) -> None:
        self.restart_requested = True

    def take_restart_request(self) -> bool:
        requested = self.restart_requested or ("r" in self.just_pressed)
        self.restart_requested = False
        return requested

    def _on_key_press(self, event: Any) -> None:
        key = self._normalize_key(event.keysym)
        self.pressed.add(key)
        self.just_pressed.add(key)

    def _on_key_release(self, event: Any) -> None:
        self.pressed.discard(self._normalize_key(event.keysym))

    @staticmethod
    def _normalize_key(keysym: str) -> str:
        aliases = {
            "Left": "left_arrow",
            "Right": "right_arrow",
            "space": "space",
            "r": "r",
            "R": "r",
            "o": "o",
            "O": "o",
            "p": "p",
            "P": "p",
        }
        return aliases.get(keysym, keysym.lower())

    def poll(self) -> bool:
        """
        Process GUI events and return False once the window is closed.

        This replaces pygame's event loop. The controller reads self.pressed,
        while the app reads self.just_pressed for one-shot actions like restart.
        """
        if self.closed:
            return False

        # just_pressed is one-frame state. Clear it before processing the new GUI
        # events for this frame.
        self.just_pressed.clear()

        try:
            self.root.update_idletasks()
            self.root.update()
            return not self.closed
        except self.tk.TclError:
            self.closed = True
            return False

    def sleep_frame(self) -> None:
        self.root.after(max(1, int(1000 / max(1, self.fps))))

    def render(self, model: Any, env: int = 0) -> None:
        s = self.scale
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.width * s, self.height * s, fill=self.COLORS[0], width=0)

        if model.owner is not None:
            grid = model.owner[env]
            ys, xs = np.nonzero(grid)
            for x, y, owner in zip(xs, ys, grid[ys, xs], strict=False):
                color = self.COLORS[int(owner)] if owner < len(self.COLORS) else self.COLORS[-1]
                self._cell(x, y, color)
        else:
            ys, xs = np.nonzero(model.occupied[env])
            for x, y in zip(xs, ys, strict=False):
                self._cell(x, y, self.COLORS[-1])

        for p in range(model.players):
            x, y = model.pos[env, p]
            if model.alive[env, p]:
                self._cell(int(x), int(y), self.COLORS[p + 1])
                self._head_marker(int(x), int(y), "#ffffff")
            else:
                self._crash_marker(int(x), int(y))

        self._draw_status(model, env)

        # --- Observation display (only for live models that support observe_lite) ---
        if hasattr(model, "observe_lite"):
            # Handle toggling and player cycling via just_pressed keys
            if "o" in self.just_pressed:
                self.show_obs = not self.show_obs
                if self.show_obs:
                    self.obs_frame.pack(fill="x", padx=8, pady=4, before=self.status_label)
                else:
                    self.obs_frame.pack_forget()

            if self.show_obs and "p" in self.just_pressed:
                self.obs_player = (self.obs_player + 1) % model.players

            if self.show_obs:
                try:
                    # observe_lite returns [envs, players, features]
                    obs_lite = model.observe_lite()[env, self.obs_player]  # shape (features,)
                    self._update_obs_display(obs_lite, self.obs_player, model.players)
                except Exception as e:
                    self.obs_text.delete(1.0, "end")
                    self.obs_text.insert("end", f"Error computing observation:\n{e}")
        else:
            # Not a live model (e.g., replay), hide the panel if it's visible
            if self.obs_frame.winfo_ismapped():
                self.obs_frame.pack_forget()

        self.root.update_idletasks()
        self.sleep_frame()

    def _update_obs_display(self, obs: np.ndarray, player_idx: int, total_players: int) -> None:
        """Format and show the observe_lite vector for a single player."""
        # obs shape: (features,)
        # Let's split into logical parts for readability
        # Features: distances(3) + xy(2) + heading_oh(4) + alive(1) + rel(7*(players-1))
        dist = obs[0:3]
        xy = obs[3:5]
        heading_oh = obs[5:9]
        alive = obs[9]
        rel = obs[10:]  # variable length

        lines = []
        lines.append(f"Player {player_idx+1} / {total_players}")
        lines.append(f"Distances (straight/left/right): {dist[0]:.3f}  {dist[1]:.3f}  {dist[2]:.3f}")
        lines.append(f"Normalized position: x={xy[0]:.3f}  y={xy[1]:.3f}")
        heading_dir = ["Up", "Right", "Down", "Left"][np.argmax(heading_oh)] if np.max(heading_oh) > 0 else "Unknown"
        lines.append(f"Heading: {heading_dir}  one-hot: {heading_oh}")
        lines.append(f"Alive: {bool(alive)}")

        # Relative features: each other player contributes 7 values
        other_players = total_players - 1
        if other_players > 0 and len(rel) == 7 * other_players:
            lines.append("Relative to others:")
            for j in range(other_players):
                start = j * 7
                rel_dx = rel[start]
                rel_dy = rel[start+1]
                rel_alive = rel[start+2]
                rel_heading = rel[start+3:start+7]
                lines.append(f"  vs P{player_idx+2+j if player_idx+2+j <= total_players else (player_idx+2+j - total_players)}: "
                             f"dxy=({rel_dx:.3f},{rel_dy:.3f}) alive={bool(rel_alive)} heading={rel_heading}")
        else:
            lines.append("Relative features: (none or unexpected length)")

        self.obs_text.delete(1.0, "end")
        self.obs_text.insert("end", "\n".join(lines))

    def render_replay(self, replay: Replay, frame: int) -> None:
        s = self.scale
        frame = int(np.clip(frame, 0, len(replay) - 1))
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, replay.width * s, replay.height * s, fill=self.COLORS[0], width=0)

        grid = replay.owner_frames[frame]
        ys, xs = np.nonzero(grid)
        for x, y, owner in zip(xs, ys, grid[ys, xs], strict=False):
            color = self.COLORS[int(owner)] if owner < len(self.COLORS) else self.COLORS[-1]
            self._cell(x, y, color)

        heads = replay.head_frames[frame]
        alive = replay.alive_frames[frame]
        for p in range(replay.players):
            x, y = heads[p]
            if alive[p]:
                self._head_marker(int(x), int(y), "#ffffff")
            else:
                self._crash_marker(int(x), int(y))

        self._draw_replay_status(replay, frame)

        # For replay mode we cannot show observe_lite, so hide the panel if it was visible
        if self.obs_frame.winfo_ismapped():
            self.obs_frame.pack_forget()

        self.root.update_idletasks()
        self.sleep_frame()

    def _draw_status(self, model: Any, env: int) -> None:
        alive_players = np.flatnonzero(model.alive[env])
        dead_players = np.flatnonzero(~model.alive[env])

        if model.done[env]:
            if len(alive_players) == 1:
                winner = int(alive_players[0])
                losers = ", ".join(f"P{int(p) + 1}" for p in dead_players) or "none"
                text = f"GAME OVER - P{winner + 1} WINS. Losers: {losers}. Press R or Restart game."
                color = self.COLORS[winner + 1]
            else:
                text = "GAME OVER - DRAW. Press R or Restart game."
                color = "#ffffff"
            self.status.set(text)
            self._banner(text, color)
        else:
            alive_text = ", ".join(f"P{int(p) + 1}" for p in alive_players)
            out_text = ", ".join(f"P{int(p) + 1}" for p in dead_players) or "none"
            self.status.set(f"Alive: {alive_text} | Eliminated: {out_text}")

    def _draw_replay_status(self, replay: Replay, frame: int) -> None:
        alive = replay.alive_frames[frame]
        alive_players = np.flatnonzero(alive)
        dead_players = np.flatnonzero(~alive)
        is_final = frame == len(replay) - 1

        if is_final:
            if len(alive_players) == 1:
                winner = int(alive_players[0])
                losers = ", ".join(f"P{int(p) + 1}" for p in dead_players) or "none"
                text = f"REPLAY END - P{winner + 1} WON. Losers: {losers}. R restarts replay."
                color = self.COLORS[winner + 1]
            else:
                text = "REPLAY END - DRAW. R restarts replay."
                color = "#ffffff"
            self.status.set(text)
            self._banner(text, color)
        else:
            alive_text = ", ".join(f"P{int(p) + 1}" for p in alive_players)
            out_text = ", ".join(f"P{int(p) + 1}" for p in dead_players) or "none"
            self.status.set(f"Replay frame {frame + 1}/{len(replay)} | Alive: {alive_text} | Eliminated: {out_text}")

    def _banner(self, text: str, color: str) -> None:
        s = self.scale
        w = self.width * s
        h = self.height * s
        self.canvas.create_rectangle(0, h // 2 - 34, w, h // 2 + 34, fill="#000000", width=0)
        self.canvas.create_text(w // 2, h // 2, text=text, fill=color, font=("Arial", 16, "bold"), width=w - 20)

    def _cell(self, x: int, y: int, color: str) -> None:
        s = self.scale
        self.canvas.create_rectangle(x * s, y * s, (x + 1) * s, (y + 1) * s, fill=color, width=0)

    def _head_marker(self, x: int, y: int, color: str) -> None:
        s = self.scale
        pad = max(1, s // 3)
        self.canvas.create_rectangle(
            x * s + pad,
            y * s + pad,
            (x + 1) * s - pad,
            (y + 1) * s - pad,
            fill=color,
            width=0,
        )

    def _crash_marker(self, x: int, y: int) -> None:
        s = self.scale
        pad = max(2, s // 5)
        self.canvas.create_line(
            x * s + pad,
            y * s + pad,
            (x + 1) * s - pad,
            (y + 1) * s - pad,
            fill="#ffffff",
            width=3,
        )
        self.canvas.create_line(
            (x + 1) * s - pad,
            y * s + pad,
            x * s + pad,
            (y + 1) * s - pad,
            fill="#ffffff",
            width=3,
        )

    def close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass