"""Fixed-ring attribute-transform/control PPO adaptation of Transform2Act.

Independent implementation of the stage-separated learning structure described
at https://github.com/Khrylx/Transform2Act. Not the original variable-topology
benchmark implementation. No skeletal transformations are possible in this task.
Geometry and the 56D physical state come from the frozen common environment.
"""
from pathlib import Path
import tempfile

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from stable_baselines3.common.policies import ActorCriticPolicy
from paper_env import PaperClimbEnv, build_model, SCENES


class TransformControlEnv(gym.Env):
    """One zero-reward design decision, then 225 physical control decisions.

    A failed design terminates with zero reward, as in the reference environment.
    Geometry rejection is recorded separately from physical climbing outcomes.
    Design decisions do not consume simulated time. All decisions still consume
    PPO transitions; both counters must be reported for compute comparisons.
    """
    metadata = {'render_modes': []}

    def __init__(self, scene, nominal_xml, seed_base, geometry_seeds=(), journal=None):
        if scene not in SCENES:
            raise ValueError(scene)
        self.scene = scene
        self.nominal = PaperClimbEnv(nominal_xml, seed_base=seed_base)
        self.core = self.nominal
        self.seed_base = int(seed_base)
        self.geometry_seeds = tuple(map(int, geometry_seeds))
        self.journal = journal
        self.episode = 0
        self.total_design_decisions = 0
        self.total_control_steps = 0
        self.total_invalid_designs = 0
        self.observation_space = gym.spaces.Dict({
            'nodes': gym.spaces.Box(-np.inf, np.inf, (8, 8), np.float32),
            'phase': gym.spaces.Box(0., 1., (1,), np.float32),
        })
        self.action_space = gym.spaces.Box(-1., 1., (12,), np.float32)
        self.scales = np.ones(8)
        self.phase = 0
        self.done = True

    def observation(self, physical):
        # Per node: angle, x, z, vx, vz, contact, contact height, length scale.
        nodes = np.column_stack((physical[:8], physical[8:24].reshape(8, 2),
                                 physical[24:40].reshape(8, 2), physical[40:48],
                                 physical[48:56], self.scales))
        return {'nodes': nodes.astype(np.float32),
                'phase': np.asarray([self.phase], np.float32)}

    @staticmethod
    def invalid_reset(core):
        return any(c.dist < -.001 and
                   (int(c.geom1) in core.env_ids or int(c.geom2) in core.env_ids)
                   for c in core.data.contact)

    def reset(self, *, seed=None, options=None):
        self.physical_seed = int(seed) if seed is not None else self.seed_base + self.episode % 32
        super().reset(seed=self.physical_seed)
        self.episode += 1
        if self.core is not self.nominal:
            self.core.close()
        self.core = self.nominal
        self.scales = np.ones(8)
        self.phase = 0
        self.done = False
        obs, info = self.core.reset(seed=self.physical_seed)
        return self.observation(obs), info

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset required after terminal transition')
        action = np.asarray(action, float)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('Expected finite 12D stage-masked action')
        inactive = action[8:] if self.phase == 0 else action[:8]
        if np.any(np.abs(inactive) > 1e-7):
            raise ValueError('Inactive stage action must be zero')
        if self.phase == 0:
            self.total_design_decisions += 1
            self.scales = 1. + .1 * np.clip(action[:8], -1, 1)
            candidate = None
            rejection = None
            try:
                # Use the identical frozen geometry compiler, not a second
                # approximate implementation. Temporary XMLs are auto-removed.
                with tempfile.TemporaryDirectory(prefix='t2a_mjcf_') as tmp:
                    path = Path(tmp) / 'candidate.xml'
                    build_model(path, self.scene, self.scales)
                    candidate = PaperClimbEnv(path, seed_base=self.seed_base)
                for seed in self.geometry_seeds:
                    candidate.reset(seed=seed)
                    if self.invalid_reset(candidate):
                        rejection = 'common_geometry_seed_penetration'
                        break
                obs, _ = candidate.reset(seed=self.physical_seed)
                if self.invalid_reset(candidate):
                    rejection = 'episode_reset_penetration'
            except ValueError:
                rejection = 'linkage_geometry_rejected'
            # Unexpected compilation/software errors are not fabricated failures.
            if rejection is not None:
                if candidate is not None:
                    candidate.close()
                self.total_invalid_designs += 1
                self.done = True
                obs, _ = self.nominal.observe()
            else:
                self.core = candidate
                self.phase = 1
            info = dict(stage='attribute', design_valid=rejection is None,
                        rejection=rejection, physical_seed=self.physical_seed,
                        rod_length_scales=self.scales.tolist(), simulation_steps=0,
                        total_control_steps=self.total_control_steps,
                        total_design_decisions=self.total_design_decisions)
            if self.done:
                info.update(completion_ratio=0., sampled_height_hold_success=False,
                            episode_return=0., invalid_design=True)
            if self.journal is not None:
                self.journal(info)
            return self.observation(obs), 0., self.done, False, info
        obs, reward, terminated, truncated, info = self.core.step(action[8:])
        self.total_control_steps += 1
        self.done = terminated or truncated
        info.update(stage='execution', design_valid=True, simulation_steps=self.core.action_repeat,
                    total_control_steps=self.total_control_steps,
                    total_design_decisions=self.total_design_decisions)
        if self.done:
            info['rod_length_scales'] = self.scales.tolist()
            if self.journal is not None:
                self.journal(info)
        return self.observation(obs), reward, terminated, truncated, info

    def close(self):
        self.core.close()
        if self.core is not self.nominal:
            self.nominal.close()


class RingNetwork(nn.Module):
    """Fixed-cycle message passing with separate self and neighbor transforms."""
    def __init__(self, inputs, hidden=64):
        super().__init__()
        self.input = nn.Linear(inputs, hidden)
        self.own = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.neighbor = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(3)])
        adjacency = torch.zeros(8, 8)
        for i in range(8):
            adjacency[i, (i - 1) % 8] = .5
            adjacency[i, (i + 1) % 8] = .5
        self.register_buffer('adjacency', adjacency)

    def forward(self, x):
        x = torch.tanh(self.input(x))
        for own, neighbor in zip(self.own, self.neighbor):
            x = torch.tanh(own(x) + neighbor(torch.matmul(self.adjacency, x)))
        return x


class MaskedGaussian:
    """A Gaussian on only the active phase's coordinates.

    Inactive coordinates carry neither probability nor entropy. SB3 retains the
    unclipped Gaussian sample for PPO likelihoods and clips only the env action.
    """
    def __init__(self, mean, log_std, phase):
        self.distribution = Normal(mean, torch.exp(log_std).expand_as(mean))
        design = (phase.reshape(-1, 1) < .5).to(mean.dtype)
        self.mask = torch.cat((design.expand(-1, 8), (1 - design).expand(-1, 4)), dim=1)

    def get_actions(self, deterministic=False):
        actions = self.distribution.mean if deterministic else self.distribution.rsample()
        return actions * self.mask

    def log_prob(self, actions):
        # Replace inactive input even when called with diagnostic arbitrary data.
        active_actions = torch.where(self.mask.bool(), actions, self.distribution.mean)
        return (self.distribution.log_prob(active_actions) * self.mask).sum(dim=1)

    def entropy(self):
        return (self.distribution.entropy() * self.mask).sum(dim=1)


class TransformControlPolicy(ActorCriticPolicy):
    """Separate graph actors and a phase-aware critic, optimized in one PPO.

    The attribute actor sees current design scales and rod identity only. The
    control actor sees physical state + scales + identity. The critic mean-pools
    all nodes (an explicit adaptation from the reference's root-node readout).
    """
    def _build(self, lr_schedule):
        if self.action_space.shape != (12,) or self.use_sde:
            raise ValueError('12D stage-masked Gaussian without gSDE required')
        self.register_buffer('node_identity', torch.eye(8))
        self.register_buffer('active_nodes', torch.tensor([1, 3, 5, 7]))
        self.attr_graph = RingNetwork(9)
        self.control_graph = RingNetwork(16)
        self.critic_graph = RingNetwork(17)
        self.attr_head = nn.Linear(64, 1)
        self.control_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.full((12,), float(self.log_std_init)))
        for module in (self.attr_head, self.control_head):
            nn.init.orthogonal_(module.weight, .01)
            nn.init.zeros_(module.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def features(self, obs):
        nodes = obs['nodes'].float()
        identity = self.node_identity.expand(nodes.shape[0], -1, -1)
        return nodes, identity

    def get_distribution(self, obs):
        nodes, identity = self.features(obs)
        attr = self.attr_head(self.attr_graph(torch.cat((nodes[:, :, 7:], identity), dim=2))).squeeze(-1)
        control = self.control_head(self.control_graph(torch.cat((nodes, identity), dim=2))).squeeze(-1)
        mean = torch.cat((attr, control[:, self.active_nodes]), dim=1)
        return MaskedGaussian(mean, self.log_std, obs['phase'])

    def predict_values(self, obs):
        nodes, identity = self.features(obs)
        phase = obs['phase'].float().reshape(-1, 1, 1).expand(-1, 8, -1)
        latent = self.critic_graph(torch.cat((nodes, identity, phase), dim=2))
        return self.value_head(latent.mean(dim=1))

    def forward(self, obs, deterministic=False):
        distribution = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic)
        return actions, self.predict_values(obs), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        distribution = self.get_distribution(obs)
        return self.predict_values(obs), distribution.log_prob(actions), distribution.entropy()

