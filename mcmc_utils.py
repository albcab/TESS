import jax
import jax.numpy as jnp
import jax.random as jrnd


def inference_loop(rng, init_state, kernel, n_iter, param):
    keys = jrnd.split(rng, n_iter)
    def step(state, key):
        state, info = kernel(key, state, param)
        return state, (state, info)
    _, (states, info) = jax.lax.scan(step, init_state, keys)
    return states, info

def inference_loop0(rng, init_state, kernel, n_iter):
    keys = jrnd.split(rng, n_iter)
    def step(state, key):
        state, info = kernel(key, state)
        return state, (state, info)
    _, (states, info) = jax.lax.scan(step, init_state, keys)
    return states, info