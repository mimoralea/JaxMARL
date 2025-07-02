import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Dict, Tuple
from jaxmarl.environments.spaces import Box, Discrete

from jaxmarl.environments.mpe.simple import SimpleMPE, State, Snap
from jaxmarl.environments.mpe.default_params import (CONTINUOUS_ACT, DISCRETE_ACT,
                                                     DT, CONTACT_FORCE,
                                                     CONTACT_MARGIN, DAMPING, MAX_STEPS)


class SimpleSumoMPE(SimpleMPE):
    """Two-player sumo wrestle in a circular arena.

    The first agent to move outside the unit circle loses. Continuous
    2-D thrust controls, zero-sum terminal reward.
    """

    def __init__(self, R: float = 0.4, action_type=DISCRETE_ACT, *, random_spawn: bool = True, **kwargs):
        self.R = R
        self.random_spawn = random_spawn
        num_agents = 2
        # Optional visual landmark – a static outline ring for rendering.
        num_landmarks = 1

        # Use color names for agents instead of player_0/player_1
        # agents = [f"player_{i}" for i in range(num_agents)]
        agents = ["green", "red"]
        landmarks = ["arena"]

        # Action & observation spaces
        if action_type == DISCRETE_ACT:
            # 5 actions: no-op, up, down, right, left (handled by base SimpleMPE)
            action_spaces = {a: Discrete(5) for a in agents}
        else:
            action_spaces = {a: Box(-1.0, 1.0, (2,)) for a in agents}

        observation_spaces = {a: Box(-jnp.inf, jnp.inf, (8,)) for a in agents}

        # Entity properties
        rad = jnp.concatenate([
            jnp.full((num_agents,), 0.05),  # agents
            jnp.array([self.R]),  # ring outline matches the actual arena size
        ])
        moveable = jnp.concatenate([
            jnp.full((num_agents,), True),
            jnp.array([False]),
        ])
        collide = jnp.concatenate([
            jnp.full((num_agents,), True),
            jnp.array([False]),
        ])

        # Colours: green, red, grey
        colour = [(38, 166, 38), (166, 38, 38), (128, 128, 128)]

        # Add the enhanced physics parameters to kwargs
        physics_kwargs = kwargs.copy()
        physics_kwargs['contact_force'] = CONTACT_FORCE * 1.2
        physics_kwargs['contact_margin'] = CONTACT_MARGIN * 1.1
        physics_kwargs['damping'] = DAMPING * 0.8

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=num_landmarks,
            landmarks=landmarks,
            action_type=action_type,
            action_spaces=action_spaces,
            observation_spaces=observation_spaces,
            colour=colour,
            rad=rad,
            moveable=moveable,
            collide=collide,
            max_steps=MAX_STEPS,
            dt=DT,
            **physics_kwargs,
        )

    # ---------------------------------------------------------------------
    # Action decoding – 2-D thrust only (no communication channels).
    # ---------------------------------------------------------------------
    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _decode_continuous_action(self, a_idx: int, action: chex.Array):
        u = action * self.accel[a_idx] * self.moveable[a_idx]
        return u, jnp.zeros((0,))

    # ---------------------------------------------------------------------
    # Deterministic reset – agents start opposite on a 0.95R circle (very close to the edge).
    # ---------------------------------------------------------------------
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        # Store the original color mapping for visualization consistency
        # Green agent (player_0) and Red agent (player_1) should be in consistent positions visually
        # regardless of which agent ID is actually on which side

        if self.random_spawn:
            # Sample two independent angles and radii inside 70%–90% of arena
            key, sub1, sub2 = jax.random.split(key, 3)
            angles = jax.random.uniform(sub1, (2,), minval=0.0, maxval=2 * jnp.pi)
            radii = jax.random.uniform(sub2, (2,), minval=0.2 * self.R, maxval=0.7 * self.R)
        else:
            # Deterministic symmetric spawn (left / right) for fair evaluation
            angles = jnp.array([0.0, jnp.pi])
            radii = jnp.array([0.5 * self.R, 0.5 * self.R])

        # Generate positions
        p_pos_agents = jnp.stack([
            jnp.array([radii[i] * jnp.cos(angles[i]), radii[i] * jnp.sin(angles[i])])
            for i in range(2)
        ])

        # Optional symmetric swap only in deterministic mode to keep visual variety
        if not self.random_spawn:
            key, subkey = jax.random.split(key)
            swap_positions = jax.random.bernoulli(subkey)
            p_pos_agents = jax.lax.cond(
                swap_positions,
                lambda _: jnp.flipud(p_pos_agents),
                lambda _: p_pos_agents,
                operand=None,
            )
        p_pos = jnp.concatenate([p_pos_agents, jnp.zeros((self.num_landmarks, 2))])

        snap = Snap(p_pos=p_pos, p_vel=jnp.zeros((self.num_entities, self.dim_p)), step=0)

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, 0)),
            done=jnp.full((self.num_agents,), False),
            step=0,
            snap=snap,
        )
        return self.get_obs(state), state

    # ---------------------------------------------------------------------
    # Observations: [self_pos, self_vel, opp_pos, opp_vel]
    # ---------------------------------------------------------------------
    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        # JAX-friendly observation function that works during tracing
        # Define a dummy observation of the right shape for when we get None during tracing
        dummy_data = jnp.zeros(8)  # 2D pos + 2D vel for self and opponent
        dummy_obs_dict = {a: dummy_data for a in self.agents}

        # Create a real observation using a JAX-primitive function
        def make_real_obs(st):
            # For each agent, get positions and velocities of self and opponent
            @partial(jax.vmap, in_axes=[0, None])
            def _obs(aidx: int, p_state):
                self_pos = p_state.p_pos[aidx]
                self_vel = p_state.p_vel[aidx]
                opp_idx = 1 - aidx  # Only two agents in Sumo
                opp_pos = p_state.p_pos[opp_idx]
                opp_vel = p_state.p_vel[opp_idx]
                return jnp.concatenate([self_pos, self_vel, opp_pos, opp_vel])

            # Use vmap to process all agents
            obs_arr = _obs(self.agent_range, st)
            return {a: obs_arr[i] for i, a in enumerate(self.agents)}

        # Using JAX primitives to handle possible None state during tracing
        # We can't directly check 'state is None' in a JIT context, so we do this instead:
        return jax.lax.cond(
            jnp.array(state is not None),  # Convert Python bool to JAX array
            lambda _: make_real_obs(state),
            lambda _: dummy_obs_dict,
            operand=None
        )

    def _agent_outside(self, state: State):
        dist = jnp.linalg.norm(state.p_pos[: self.num_agents], axis=1)
        agent_radius = self.rad[: self.num_agents]
        return dist + agent_radius >= self.R

    def dones(self, state: State, next_state: State) -> Dict[str, bool]:
        outside = self._agent_outside(next_state)
        ring_done = jnp.any(outside)
        time_done = next_state.step >= self.max_steps
        dones = jnp.full((self.num_agents,), ring_done | time_done)
        return dones

    # ---------------------------------------------------------------------
    # Reward function for the sumo environment
    # ---------------------------------------------------------------------
    def rewards(self, state: State, next_state: State) -> Dict[str, float]:
        """Calculate rewards based on agents' positions relative to the arena boundary."""
        outside = self._agent_outside(next_state)
        # time_done = next_state.step >= self.max_steps
        opponent_out = jnp.roll(outside, 1)

        # win/loss conditions
        win = (~outside) & opponent_out
        # loss = outside | time_done
        loss = outside & (~opponent_out)

        rewards = jnp.zeros((self.num_agents,))
        rewards = jnp.where(win, 1.0, jnp.where(loss, -1.0, 0.0))

        rew_dict = {a: rewards[i] for i, a in enumerate(self.agents)}
        return rew_dict
