#!/usr/bin/env python3
"""
Continuous training for Tron agents – GA or RL, infinite horizon, automatic checkpoints.
Usage:
    python continuous_trainer.py --mode ga --checkpoint_steps 10 --total_generations 0
    python continuous_trainer.py --mode rl --checkpoint_steps 50000 --total_timesteps 0
"""

import argparse
import json
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from model import TronBatchModel
from controller import GreedySpaceController, RandomController
from trainer import MLPPolicy, evaluate_controller, calculate_fitness, export_genome
from reinforcement import TronSingleAgentEnv


# ----------------------------------------------------------------------
# GA specific helpers
# ----------------------------------------------------------------------
def make_child(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator,
               mutation_std: float = 0.03, mutation_rate: float = 0.05) -> np.ndarray:
    """Crossover and mutation for GA."""
    mask = rng.random(parent_a.shape) < 0.5
    child = np.where(mask, parent_a, parent_b).copy()
    mutate = rng.random(child.shape) < mutation_rate
    child[mutate] += rng.normal(0, mutation_std, size=mutate.sum())
    return child.astype(np.float32)


def evaluate_genome(genome: np.ndarray, obs_dim: int, hidden: int,
                    envs: int, width: int, height: int, players: int, seed: int) -> float:
    """Fitness: combination of win rate and survival length."""
    policy = MLPPolicy(obs_dim, hidden=hidden, genome=genome)
    scores = evaluate_controller(policy, envs=envs, width=width, height=height,
                                 players=players, seed=seed)
    return calculate_fitness(scores["win_rate_per_player"].mean(), scores["mean_length"])


class ContinuousGATrainer:
    """
    Indefinite GA training with periodic checkpoints.
    """
    def __init__(self, output_dir: Path, hidden: int, pop_size: int, elite_count: int,
                 eval_envs: int, width: int, height: int, players: int,
                 checkpoint_steps: int, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints_ga"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.hidden = hidden
        self.pop_size = pop_size
        self.elite_count = elite_count
        self.eval_envs = eval_envs
        self.width = width
        self.height = height
        self.players = players
        self.checkpoint_steps = checkpoint_steps
        self.seed = seed

        # Determine observation dimension
        probe = TronBatchModel(width=width, height=height, players=players, envs=1)
        self.obs_dim = probe.observe_lite().shape[-1]
        # Number of genome parameters
        self.n_params = MLPPolicy(self.obs_dim, hidden=hidden).n_params

        self.rng = np.random.default_rng(seed)
        self.population = None
        self.generation = 0
        self.best_fitness = -np.inf
        self.best_genome = None
        self.start_time = time.time()
        self.running = True

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\nSIGINT received – saving final checkpoint and exiting...")
        self.running = False
        self._save_checkpoint(final=True)
        sys.exit(0)

    def _save_checkpoint(self, final: bool = False):
        """Save full trainer state and best genome."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"gen_{self.generation:06d}_{timestamp}.npz"
        np.savez(
            checkpoint_path,
            population=self.population,
            generation=self.generation,
            best_fitness=self.best_fitness,
            best_genome=self.best_genome,
            rng_state=self.rng.bit_generator.state,
        )
        print(f"Saved GA checkpoint: {checkpoint_path}")

        # Also save the best genome in standard format
        if self.best_genome is not None:
            metadata = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "training_duration_seconds": time.time() - self.start_time,
                "generation": self.generation,
                "best_fitness": float(self.best_fitness),
                "eval_envs": self.eval_envs,
                "hidden": self.hidden,
                "pop_size": self.pop_size,
                "elite_count": self.elite_count,
                "width": self.width,
                "height": self.height,
                "players": self.players,
                "seed": self.seed,
            }
            genome_path, meta_path = export_genome(
                self.best_genome,
                obs_dim=self.obs_dim,
                hidden=self.hidden,
                fitness=self.best_fitness,
                generations=self.generation,
                pop_size=self.pop_size,
                elite_count=self.elite_count,
                eval_envs=self.eval_envs,
                width=self.width,
                height=self.height,
                players=self.players,
                seed=self.seed,
                out_dir=self.output_dir / "genomes",
            )
            # Augment metadata with training time
            with open(meta_path, "r") as f:
                old_meta = json.load(f)
            old_meta.update(metadata)
            with open(meta_path, "w") as f:
                json.dump(old_meta, f, indent=2)

    def _load_latest_checkpoint(self):
        """Find most recent checkpoint and resume."""
        checkpoints = sorted(self.checkpoint_dir.glob("gen_*.npz"))
        if not checkpoints:
            return False
        latest = checkpoints[-1]
        data = np.load(latest, allow_pickle=True)
        self.population = data["population"]
        self.generation = int(data["generation"]) + 1   # resume from next generation
        self.best_fitness = float(data["best_fitness"])
        self.best_genome = data["best_genome"]
        self.rng.bit_generator.state = data["rng_state"].item()
        print(f"Resumed GA from {latest.name} at generation {self.generation}")
        return True

    def _evaluate_and_log(self, genome: np.ndarray) -> Dict[str, float]:
        """Run baseline evaluation (vs greedy/random) for logging."""
        policy = MLPPolicy(self.obs_dim, hidden=self.hidden, genome=genome)

        # vs greedy
        greedy_scores = evaluate_controller(
            policy, envs=1024, width=self.width, height=self.height,
            players=self.players, seed=self.seed
        )
        # vs random
        random_scores = evaluate_controller(
            policy, envs=1024, width=self.width, height=self.height,
            players=self.players, seed=self.seed+1
        )
        return {
            "winrate_vs_greedy": greedy_scores["win_rate_per_player"].mean(),
            "winrate_vs_random": random_scores["win_rate_per_player"].mean(),
            "mean_length_vs_greedy": greedy_scores["mean_length"],
            "mean_length_vs_random": random_scores["mean_length"],
        }

    def train(self, total_generations: int = 0):
        """
        Run indefinitely if total_generations == 0, else fixed generations.
        """
        if not self._load_latest_checkpoint():
            # Initialize new population
            self.population = self.rng.normal(0, 0.2, size=(self.pop_size, self.n_params)).astype(np.float32)
            self.generation = 0

        while self.running:
            # Evaluate fitness
            fitness = np.array([
                evaluate_genome(g, self.obs_dim, self.hidden, self.eval_envs,
                                self.width, self.height, self.players, self.seed + self.generation)
                for g in self.population
            ])

            # Sort by fitness
            order = np.argsort(fitness)[::-1]
            self.population = self.population[order]
            fitness = fitness[order]

            # Update best
            if fitness[0] > self.best_fitness:
                self.best_fitness = fitness[0]
                self.best_genome = self.population[0].copy()
                # Evaluate best genome against baselines
                eval_results = self._evaluate_and_log(self.best_genome)
                print(f"[gen {self.generation}] New best fitness={self.best_fitness:.4f} "
                      f"win vs greedy={eval_results['winrate_vs_greedy']:.3f} "
                      f"vs random={eval_results['winrate_vs_random']:.3f}")

            # Print stats
            print(f"gen={self.generation:04d} best={fitness[0]:+.4f} mean={fitness.mean():+.4f}")

            # Create next generation
            elites = self.population[:self.elite_count]
            new_pop = [e.copy() for e in elites]
            while len(new_pop) < self.pop_size:
                a, b = self.rng.choice(self.elite_count, size=2, replace=True)
                child = make_child(elites[a], elites[b], self.rng)
                new_pop.append(child)
            self.population = np.stack(new_pop)

            self.generation += 1

            # Checkpoint if needed
            if self.generation % self.checkpoint_steps == 0:
                self._save_checkpoint()

            # Stop if finite generations reached
            if total_generations and self.generation >= total_generations:
                print(f"Reached total_generations={total_generations}. Stopping.")
                break


# ----------------------------------------------------------------------
# RL specific trainer (PPO)
# ----------------------------------------------------------------------
class ContinuousRLTrainer:
    """
    Indefinite PPO training with periodic model checkpoints and evaluation.
    """
    def __init__(self, output_dir: Path, width: int, height: int, players: int,
                 opponent_policy: str, checkpoint_steps: int, seed: int = 42,
                 learning_rate: float = 3e-4, n_steps: int = 2048,
                 batch_size: int = 64, n_epochs: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints_rl"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.width = width
        self.height = height
        self.players = players
        self.opponent_policy = opponent_policy
        self.checkpoint_steps = checkpoint_steps
        self.seed = seed

        self.env = TronSingleAgentEnv(
            width=width, height=height, players=players,
            opponent_policy=opponent_policy, seed=seed, render_mode=None
        )
        self.vec_env = DummyVecEnv([lambda: Monitor(self.env)])
        self.model = None
        self.total_timesteps = 0
        self.start_time = time.time()
        self.running = True

        self.ppo_kwargs = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": [256, 256]},
            "verbose": 1,
            "seed": seed,
        }

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\nSIGINT received – saving final model and exiting...")
        self.running = False
        self._save_checkpoint(final=True)
        sys.exit(0)

    def _save_checkpoint(self, final: bool = False):
        """Save PPO model and training state."""
        if self.model is None:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.checkpoint_dir / f"ppo_{self.total_timesteps:010d}_{timestamp}.zip"
        self.model.save(str(model_path))
        print(f"Saved RL checkpoint: {model_path}")

        # Also save a metadata json
        metadata = {
            "timesteps": self.total_timesteps,
            "training_duration_seconds": time.time() - self.start_time,
            "checkpoint_type": "final" if final else "periodic",
            "config": {
                "width": self.width,
                "height": self.height,
                "players": self.players,
                "opponent_policy": self.opponent_policy,
                **self.ppo_kwargs
            }
        }
        meta_path = model_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_latest_checkpoint(self) -> bool:
        """Resume from most recent .zip checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("ppo_*.zip"))
        if not checkpoints:
            return False
        latest = checkpoints[-1]
        self.model = PPO.load(str(latest))
        # Extract timesteps from filename
        parts = latest.stem.split("_")
        self.total_timesteps = int(parts[1]) if len(parts) > 1 else 0
        print(f"Resumed RL from {latest.name} at {self.total_timesteps} timesteps")
        return True

    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate current model vs greedy and random using run_match (see evaluate.py)."""
        # Import here to avoid circular imports
        from evaluate import run_match, RLPolicyWrapper
        rl_wrapper = RLPolicyWrapper(self.model, player_id=0)
        greedy_wrapper = GreedySpaceController()
        random_wrapper = RandomController(seed=self.seed)

        # vs greedy
        stats_greedy = run_match(rl_wrapper, greedy_wrapper, envs=1024,
                                 width=self.width, height=self.height, seed=self.seed)
        # vs random
        stats_random = run_match(rl_wrapper, random_wrapper, envs=1024,
                                 width=self.width, height=self.height, seed=self.seed+1)
        return {
            "winrate_vs_greedy": stats_greedy["win_rate_0"],
            "winrate_vs_random": stats_random["win_rate_0"],
            "avg_survival_vs_greedy": stats_greedy["avg_survival_0"],
            "avg_survival_vs_random": stats_random["avg_survival_0"],
        }

    def train(self, total_timesteps: int = 0):
        """
        Loop training indefinitely (if total_timesteps == 0) or fixed steps.
        """
        if not self._load_latest_checkpoint():
            # Create fresh model
            self.model = PPO("MlpPolicy", self.vec_env, **self.ppo_kwargs)
            self.total_timesteps = 0

        # Track next checkpoint
        next_checkpoint = self.total_timesteps + self.checkpoint_steps
        if total_timesteps:
            target = total_timesteps
        else:
            target = 10**12   # effectively infinite

        while self.running and self.total_timesteps < target:
            # Determine how many steps to train this chunk
            chunk = min(self.checkpoint_steps, target - self.total_timesteps)
            self.model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            self.total_timesteps += chunk

            # Evaluate and log
            eval_results = self._evaluate_model()
            print(f"[RL] timestep={self.total_timesteps:>10d}  "
                  f"win vs greedy={eval_results['winrate_vs_greedy']:.3f} "
                  f"vs random={eval_results['winrate_vs_random']:.3f}")

            # Save checkpoint
            self._save_checkpoint()

            # If infinite, continue; otherwise stop when target reached
            if total_timesteps and self.total_timesteps >= total_timesteps:
                break


# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Continuous training for Tron agents")
    parser.add_argument("--mode", required=True, choices=["ga", "rl"], help="Training method")
    parser.add_argument("--output_dir", type=str, default="training_runs", help="Root output directory")
    parser.add_argument("--checkpoint_steps", type=int, default=10,
                        help="GA: generations per checkpoint; RL: timesteps per checkpoint")
    parser.add_argument("--total_generations", type=int, default=0,
                        help="GA: max generations (0=infinite)")
    parser.add_argument("--total_timesteps", type=int, default=0,
                        help="RL: max timesteps (0=infinite)")

    # Environment parameters
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--players", type=int, default=2, choices=[2,3,4])
    parser.add_argument("--seed", type=int, default=42)

    # GA specific
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--pop_size", type=int, default=64)
    parser.add_argument("--elite_count", type=int, default=8)
    parser.add_argument("--eval_envs", type=int, default=1024)

    # RL specific
    parser.add_argument("--opponent", type=str, default="greedy", choices=["greedy", "random"])
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)

    args = parser.parse_args()

    # Create a unique run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.mode}_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save command line arguments
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.mode == "ga":
        trainer = ContinuousGATrainer(
            output_dir=run_dir,
            hidden=args.hidden,
            pop_size=args.pop_size,
            elite_count=args.elite_count,
            eval_envs=args.eval_envs,
            width=args.width,
            height=args.height,
            players=args.players,
            checkpoint_steps=args.checkpoint_steps,
            seed=args.seed,
        )
        trainer.train(total_generations=args.total_generations)

    else:  # rl
        trainer = ContinuousRLTrainer(
            output_dir=run_dir,
            width=args.width,
            height=args.height,
            players=args.players,
            opponent_policy=args.opponent,
            checkpoint_steps=args.checkpoint_steps,
            seed=args.seed,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
        )
        trainer.train(total_timesteps=args.total_timesteps)

    print("Training finished or interrupted.")


if __name__ == "__main__":
    main()