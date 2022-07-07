from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.flatten_util import ravel_pytree


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


def stein_disc(X, logprob_fn, beta=-1/2) -> Tuple:
    """Stein discrepancy with inverse multi-quadric kernel,
    i.e. (1 + (x - x')T(x - x')) ** beta
    returns U-Statistic (unbiased) and V-statistic (biased)
    """

    X = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), X)
    if isinstance(X, dict):
        d = 0
        for name, x in X.items():
            s = x.shape
            if not s[1:]:
                d += 1
            else:
                d += s[1]
        T = s[0]
        sub = lambda x, x_: ravel_pytree(x)[0] - ravel_pytree(x_)[0]
        grad = lambda x: ravel_pytree(jax.grad(logprob_fn)(x))[0]
    else: 
        T, d = X.shape
        sub = lambda x, x_: x - x_
        grad = jax.grad(logprob_fn)
    beta = -beta

    # gradk = lambda diff, dot_prod: -2 * beta * (1 + dot_prod) ** (-beta - 1) * diff
    # gradgradk = lambda diff, dot_prod: -2 * beta * jnp.sum(-diff ** 2 * 2 * (-beta - 1) * (1 + dot_prod) ** (-beta - 2) - (1 + dot_prod) ** (-beta - 1))

    # def disc2(x, x_):
    #     diff = sub(x, x_)
    #     dot_prod = jnp.dot(diff, diff)
    #     dx = grad(x)
    #     dx_ = grad(x_)
    #     return (
    #         jnp.dot(dx, dx_) * (1 + dot_prod) ** (-beta)
    #         + jnp.dot(dx, -gradk(diff, dot_prod)) + jnp.dot(dx_, gradk(diff, dot_prod))
    #         + gradgradk(diff, dot_prod)
    #     )

    def disc(x, x_):
        diff = sub(x, x_)
        dot_prod = jnp.dot(diff, diff)
        dx = grad(x)
        dx_ = grad(x_)
        return (
            -4 * beta * (beta+1) * dot_prod / ((1 + dot_prod) ** (beta + 2))
            + 2 * beta * (d + jnp.dot(dx - dx_, diff)) / ((1 + dot_prod) ** (1 + beta))
            + jnp.dot(dx, dx_) / ((1 + dot_prod) ** beta)
        )

    _disc = jax.vmap(disc, (None, 0))
    # _disc2 = jax.vmap(disc2, (None, 0))
    # try:
    #     mc_sum = jax.vmap(_disc, (0, None))(X, X).sum()
    #     mc_sum2 = jax.vmap(_disc2, (0, None))(X, X).sum()
    #     print(mc_sum, mc_sum2)
    # except RuntimeError:
    mc_sum = jax.lax.map(lambda x: _disc(x, X).sum(), X).sum()
    return (mc_sum - jax.vmap(lambda x: disc(x, x))(X).sum()) / (T * (T-1)), mc_sum / T**2