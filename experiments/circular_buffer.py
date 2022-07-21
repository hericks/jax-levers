from typing import Tuple
from functools import partial

import chex
import jax
import jax.numpy as jnp

from flax import struct


@struct.dataclass
class CircularBufferState:
    s_mem: chex.Array
    a_mem: chex.Array
    ns_mem: chex.Array
    r_mem: chex.Array
    done_mem: chex.Array
    index: int
    n_elements: int


def circular_buffer_reset(
    capacity: int, dummy: chex.Array
) -> CircularBufferState:
    buffer_state = CircularBufferState(
        s_mem=jnp.zeros((capacity, *dummy.shape)),
        a_mem=jnp.zeros((capacity)),
        ns_mem=jnp.zeros((capacity, *dummy.shape)),
        r_mem=jnp.zeros((capacity)),
        done_mem=jnp.full((capacity), False),
        index=0,
        n_elements=0
    )
    return buffer_state


@jax.jit
def circular_buffer_push(
    state: CircularBufferState,
    s: chex.Array,
    a: int,
    ns: chex.Array,
    r: float,
    done: bool
) -> CircularBufferState:
    n_state = CircularBufferState(
        s_mem=state.s_mem.at[state.index,:].set(s),
        a_mem=state.a_mem.at[state.index].set(a),
        ns_mem=state.ns_mem.at[state.index,:].set(ns),
        r_mem=state.r_mem.at[state.index].set(r),
        done_mem=state.done_mem.at[state.index].set(done),
        index=(state.index+1)%(state.s_mem.shape[0]),
        n_elements= jnp.maximum(state.index+1, state.n_elements)
    )
    return n_state

@partial(jax.jit, static_argnums=(2,))
def circular_buffer_sample(
    rng: chex.PRNGKey, state: CircularBufferState, batch_size: int
) -> Tuple[chex.Array, int, chex.Array, float, bool]:
    idx = jax.random.randint(rng, (batch_size,), 0, state.n_elements)
    return (state.s_mem[idx,:],
            state.a_mem[idx],
            state.ns_mem[idx,:],
            state.r_mem[idx],
            state.done_mem[idx])