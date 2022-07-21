from functools import partial

import jax 
import jax.numpy as jnp
import numpy as np
import gymnax
import optax
import flax.linen as nn

from circular_buffer import *

from jax.config import config
config.update('jax_disable_jit', True)

# Network model
class MLP(nn.Module):
    """Simple ReLU MLP."""

    num_hidden_units: int
    num_hidden_layers: int
    num_output_units: int

    @nn.compact
    def __call__(self, x):
        for l in range(self.num_hidden_layers):
            x = nn.Dense(features=self.num_hidden_units)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.num_output_units)(x)
        return x

rng = jax.random.PRNGKey(42)
rng, rng_reset, rng_policy, rng_step = jax.random.split(rng, 4)

# Create and reset the environment
env, env_params = gymnax.make("MountainCar-v0")
obs, state = env.reset(rng_reset, env_params)

# Obtain important parameters of the environment
n_actions = env.action_space(env_params).n
obs_dim = obs.shape[0]

# Create model for q- and target-network
# initialize model parameters
model = MLP(4, 2, n_actions)
network_params = model.init(rng, jnp.zeros(obs_dim))

@partial(jax.jit, static_argnames=['capacity', 'steps_in_episode', 'batch_size'])
def rollout(
    rng_input, 
    policy_params, 
    capacity, 
    batch_size,
    gamma,
    steps_between_target_updates,
    epsilon,
    env_params, 
    steps_in_episode
):
    """Rollout a jitted gymnax episode with lax.scan."""
    # Reset the environment
    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, state = env.reset(rng_reset, env_params)

    # Reset the learning agent
    # - experience buffer
    buffer_state = circular_buffer_reset(capacity, obs)
    # - network parameters
    target_params = policy_params
    steps_until_update = steps_between_target_updates
    # - optimizer
    opt = optax.adam(learning_rate=0.001)
    opt_state = opt.init(policy_params)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, policy_params, target_params, buffer_state, opt_state, steps_until_update, rng = state_input
        rng, rng_epsilon, rng_action, rng_step = jax.random.split(rng, 4)
        
        # choose action via epsilon greedy
        action = jax.lax.cond(
            jax.random.uniform(rng_epsilon) <= epsilon,
            lambda x: env.action_space(env_params).sample(rng_action),
            lambda x: jnp.argmax(model.apply(policy_params, obs)),
            None
        )
        next_obs, next_state, reward, done, _ = env.step(
          rng_step, state, action, env_params
        )
        
        # update experience replay
        buffer_state = circular_buffer_push(
            buffer_state, 
            obs,
            action,
            next_obs,
            reward,
            done
        )

        # update target network if necessary
        # TODO: should be based on the number of ROUNDS
        target_params = jax.lax.cond(
            steps_until_update == 0,
            lambda: policy_params,
            lambda: target_params,
        )

        steps_unitl_update = jax.lax.cond(
            steps_until_update == 0,
            lambda: steps_between_target_updates,
            lambda: steps_until_update - 1
        )

        # compute loss and update q-network
        rng, rng_policy_update = jax.random.split(rng, 2)

        def update_policy_params(
            rng, buffer_state, policy_params, target_params, opt_state
        ):
            # sample from experience replay
            states, actions, next_states, rewards, dones = \
                circular_buffer_sample(rng, buffer_state, batch_size)

            def q_loss(policy_params):
                # compute current estimates of action state values
                state_action_vals = model.apply(
                    policy_params, states
                )[np.arange(len(states)), actions.astype(int)]
                # next_state_value != 0 only for non-terminal next states
                # TODO: Avoid compiting policy for all next states while staying
                #       jit compatible.
                next_state_vals = jnp.where(
                    ~dones, 
                    model.apply(target_params, next_states).max(1),
                    0
                )
                expected_state_action_vals = gamma * next_state_vals + rewards
                diff = state_action_vals - expected_state_action_vals
                return jnp.inner(diff, diff) / 2.0

            loss, grads = jax.value_and_grad(q_loss)(policy_params)
            print(loss)
            updates, updated_opt_state = opt.update(grads, opt_state)
            updated_policy_params = optax.apply_updates(policy_params, updates)
            return updated_policy_params, updated_opt_state

        policy_params, opt_state = jax.lax.cond(
            batch_size <= buffer_state.n_elements,
            update_policy_params,
            lambda *args: (policy_params, opt_state),
            rng_policy_update, buffer_state, policy_params, target_params, opt_state
        )

        carry = [
            next_obs, next_state, policy_params, target_params, buffer_state,
            opt_state, steps_until_update, rng
        ]
        return carry, [obs, action, reward, next_obs, done]

    # Scan over episode step loop
    initial_scan_state = [
        obs, state,                                         # environment
        policy_params, target_params, buffer_state,         # policy
        opt_state,                                          # optimizer
        steps_until_update,                                 
        rng_episode                                         # rng
    ]
    _, scan_out = jax.lax.scan(
      policy_step,
      initial_scan_state,
      (),
      steps_in_episode
    )

    # Return masked sum of rewards accumulated by agent in episode
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done

# epsilon-greedy
epsilon = 0.

# replay-buffer capacity
capacity = 16
batch_size = 8
gamma = 1.0

# how often to update target update
steps_between_target_updates = 10

steps_per_episode = 200
num_episodes = 100
num_steps = steps_per_episode * num_episodes

import time

prev = time.time()
obs, action, reward, next_obs, done = rollout(
    rng, 
    network_params,
    capacity,
    batch_size,
    gamma,
    steps_between_target_updates,
    epsilon,
    env_params,
    num_steps
)
print(f'RUN: {time.time()-prev}')

prev = time.time()
obs, action, reward, next_obs, done = rollout(
    rng, 
    network_params,
    capacity,
    batch_size,
    gamma,
    steps_between_target_updates,
    epsilon,
    env_params,
    num_steps
)
print(f'RUN: {time.time()-prev}')

print(obs.shape, reward.shape, jnp.sum(reward), done.shape, jnp.sum(done))