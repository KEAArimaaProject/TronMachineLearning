from abc import ABC, abstractmethod
import numpy as np

class TronAgent(ABC):
    """Abstract agent for the Tron batch environment."""

    @property
    @abstractmethod
    def observation_type(self) -> str:
        """'lite' or 'grid' – tells the runner what observation to give this agent."""
        ...

    @abstractmethod
    def act(
        self,
        observation: np.ndarray,      # float, shape [envs, ...] depending on type
        legal_actions: np.ndarray,    # bool, [envs, 3]
    ) -> np.ndarray:
        """Return actions for all envs, shape [envs], int in {0,1,2}."""
        ...



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

    def sample_actions(self, model, temperature=0.5):
        observations = model.observe_lite()
        logits = self.logits(observations)  # (envs, players, 3)
        probs = np.exp(logits / temperature)
        probs /= probs.sum(axis=-1, keepdims=True)
        # Sample action for each env and player
        actions = np.apply_along_axis(
            lambda p: np.random.choice(3, p=p), -1, probs
        ).astype(np.int8)
        return actions

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

