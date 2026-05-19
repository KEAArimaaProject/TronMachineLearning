from datetime import datetime
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.model import TronBatchModel
from src.view import GameView


def run_ga(
    generations=20,
    pop_size=40,
    elite_count=6,
    hidden=32,
    eval_envs=1024,

    players=2,
    width=32,
    height=32,
    seed=0,
    opponent=None,
):
    rng = np.random.default_rng(seed)

    environment_variables = TronBatchModel(width=width, height=height, players=players, envs=1)
    found_parameters = environment_variables.observe_lite().shape[-1]
    n_params = MLPPolicy(found_parameters, hidden=hidden).n_params

    population = rng.normal(0, 0.2, size=(pop_size, n_params)).astype(np.float32)
    best = None

    for gen in range(generations):
        if opponent is None:
            fitness = np.array([
                evaluate_genome(g, found_parameters, hidden=hidden, envs=eval_envs, seed=seed + gen)
                for g in population
            ])
        else:
            fitness = np.array([
                evaluate_genome_vs_opponent(g, found_parameters, opponent, hidden=hidden, envs=eval_envs, seed=seed + gen, players=players)
                for g in population
            ])

        order = np.argsort(fitness)[::-1]
        population = population[order]
        fitness = fitness[order]

        if best is None or fitness[0] > best[0]:
            best = (float(fitness[0]), population[0].copy())

        print(f"gen={gen:03d} best={fitness[0]:+.4f} mean={fitness.mean():+.4f}")

        elites = population[:elite_count]
        new_pop = [e.copy() for e in elites]

        while len(new_pop) < pop_size:
            a, b = rng.choice(elite_count, size=2, replace=True)
            new_pop.append(make_child(elites[a], elites[b], rng))

        population = np.stack(new_pop)

    _best_fitness, _best_genome = best
    return _best_fitness, _best_genome, found_parameters

def make_child(parent_a, parent_b, rng, mutation_std=0.03, mutation_rate=0.05):
    mask = rng.random(parent_a.shape) < 0.5
    child = np.where(mask, parent_a, parent_b).copy()

    mutate = rng.random(child.shape) < mutation_rate
    child[mutate] += rng.normal(0, mutation_std, size=mutate.sum())
    return child.astype(np.float32)

def evaluate_genome(genome, obs_dim, *, hidden=32, envs=1024, seed=0):
    def _make_controller_from_arg(arg):
        # If arg is a controller class, instantiate it. If it's already an
        # instance, return it.
        if arg is None:
            return None
        if isinstance(arg, type):
            return arg()
        return arg

    _policy = MLPPolicy(obs_dim, hidden=hidden, genome=genome)
    # Default: policy controls all players (symmetric self-play)
    scores = evaluate_controller(_policy, envs=envs, seed=seed)
    return float(
        0.25 * scores["win_rate_per_player"].mean()
        + 0.001 * scores["mean_length"]
    )


def evaluate_genome_vs_opponent(genome, obs_dim, opponent, *, hidden=32, envs=1024, seed=0, players=2):
    """Evaluate a genome playing as player 0 against `opponent` for other players.

    `opponent` may be a controller instance or a controller class (which will be instantiated).
    """
    _policy = MLPPolicy(obs_dim, hidden=hidden, genome=genome)
    if isinstance(opponent, type):
        opp_inst = opponent()
    else:
        opp_inst = opponent

    # Build per-player controllers: policy for player 0, opponent for others
    controllers = [None] * players
    for p in range(players):
        controllers[p] = _policy if p == 0 else opp_inst

    scores = evaluate_controller(controllers, envs=envs, seed=seed, players=players)
    return float(
        0.25 * scores["win_rate_per_player"].mean()
        + 0.001 * scores["mean_length"]
    )


def evaluate_controller(controller, *, envs=4096, width=32, height=32, players=2, max_ticks=512, seed=0):
    env = TronBatchModel(
        width=width,
        height=height,
        players=players,
        envs=envs,
        keep_owner=False,
        seed=seed,
    )

    total_reward = np.zeros((envs, players), dtype=np.float32)
    wins = np.zeros((envs, players), dtype=np.float32)
    lengths = np.zeros(envs, dtype=np.int32)

    for _ in range(max_ticks):
        # Support either a single controller (controls all players) or
        # a sequence of per-player controllers. If a sequence is provided
        # we call each controller and pick its column for the matching
        # player index. To avoid duplicate calls we cache results per
        # controller id.
        if isinstance(controller, (list, tuple)):
            actions = np.zeros((envs, players), dtype=np.int8)
            cache = {}
            for p, ctrl in enumerate(controller):
                cid = id(ctrl)
                if cid in cache:
                    res = cache[cid]
                else:
                    res = ctrl.actions(env)
                    cache[cid] = res
                # Take the column corresponding to player p
                actions[:, p] = res[:, p]
        else:
            actions = controller.actions(env)
        step = env.step(actions)
        total_reward += step.reward

        active = ~step.done
        lengths[active] += 1

        newly_finished = step.done
        if newly_finished.any():
            alive = step.alive[newly_finished]
            winner_mask = alive.sum(axis=1) == 1
            if winner_mask.any():
                finished_ids = np.flatnonzero(newly_finished)
                winner_ids = finished_ids[winner_mask]
                winners = alive[winner_mask].argmax(axis=1)
                wins[winner_ids, winners] = 1.0

            env.auto_reset_done()

    return {
        "mean_reward_per_player": total_reward.mean(axis=0),
        "win_rate_per_player": wins.mean(axis=0),
        "mean_length": lengths.mean(),
    }

def watch_policy(policy, players=2, width=32, height=32, scale=16, fps=20, seed=0):
    model = TronBatchModel(width=width, height=height, players=players, envs=1, keep_owner=True, seed=seed)
    view = GameView(model, scale=scale, fps=fps)
    try:
        while view.poll():
            if view.take_restart_request():
                model.reset()
            if not model.done[0]:
                model.step(policy.actions(model))
            view.render(model)
    finally:
        view.close()

def watch_mixed_policy(policies, players=2, width=32, height=32, scale=16, fps=20, seed=0):
    board_model = TronBatchModel(width=width, height=height, players=players, envs=1, keep_owner=True, seed=seed)
    view = GameView(board_model, scale=scale, fps=fps)

    try:
        while view.poll():
            if view.take_restart_request():
                board_model.reset()
            if not board_model.done[0]:
                acts = np.zeros((1, board_model.players), dtype=np.int8)
                for i in range(len(policies)):
                    policy = policies[i]

                    policy_acts = policy.actions(board_model)
                    acts[:, i] = policy_acts[:, i]
                board_model.step(acts)
            view.render(board_model)
    finally:
        view.close()




def export_genome(
    genome,
    *,
    obs_dim,
    hidden,
    fitness,
    generations,
    pop_size,
    elite_count,
    eval_envs,
    width,
    height,
    players=2,
    seed,
    out_dir=Path("tron_genomes"),
):
    """
    Save a genome and metadata into ./tron_genomes/.

    Files created:
        tron_genomes/<name>.npy
        tron_genomes/<name>.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    name = (
        f"tron_"
        f"{timestamp}"
    )

    genome_path = out_dir / f"{name}.npy"
    metadata_path = out_dir / f"{name}.json"

    np.save(genome_path, np.asarray(genome, dtype=np.float32))

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "genome_file": genome_path.name,
        "fitness": None if fitness is None else float(fitness),
        "model": {
            "policy_type": "MLPPolicy",
            "obs_dim": int(obs_dim),
            "hidden": int(hidden),
            "action_count": 3,
        },
        "environment": {
            "width": int(width),
            "height": int(height),
            "players": int(players),
        },
        "genetic_algorithm": {
            "generations": int(generations),
            "pop_size": int(pop_size),
            "elite_count": int(elite_count),
            "eval_envs": int(eval_envs),
            "seed": int(seed),
        },
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved genome:  {genome_path}")
    print(f"Saved metadata: {metadata_path}")

    return genome_path, metadata_path


# REGION CLases
class MLPPolicy:
    def __init__(self, obs_dim, hidden=32, genome=None, rng=None):
        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)
        self.rng = np.random.default_rng(rng)

        self.n_params = (
            self.obs_dim * self.hidden + self.hidden +
            self.hidden * 3 + 3
        )

        if genome is None:
            genome = self.rng.normal(0, 0.1, size=self.n_params).astype(np.float32)
        else:
            genome = np.asarray(genome, dtype=np.float32)
        self.genome = genome
        self._unpack_genome(genome)

    def _unpack_genome(self, genome):
        assert genome.shape == (self.n_params,)
        i = 0
        n = self.obs_dim * self.hidden
        self.w1 = genome[i:i+n].reshape(self.obs_dim, self.hidden)
        i += n

        self.b1 = genome[i:i+self.hidden]
        i += self.hidden

        n = self.hidden * 3
        self.w2 = genome[i:i+n].reshape(self.hidden, 3)
        i += n

        self.b2 = genome[i:i+3]

    def set_genome(self, genome):
        genome = np.asarray(genome, dtype=np.float32)
        self.genome = genome
        self._unpack_genome(genome)

    def get_genome(self):
        return self.genome

    def logits(self, observations):
        # obs: [envs, players, obs_dim]
        h = np.tanh(observations @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def actions(self, model):
        observations = model.observe_lite()
        logits = self.logits(observations)
        return logits.argmax(axis=-1).astype(np.int8)

# ENDREGION CLasses