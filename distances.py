from typing import Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class kullback_liebler:
    def __new__(cls, logprob_fn, d, flow, flow_inv) -> Tuple:

        def reverse_kld(param, u, k):
            v = jax.random.normal(k, (d,))
            x, v_, ldj = flow(u, v, param, k)
            u = ravel_pytree(u)[0]
            return (
                -.5 * jnp.dot(u, u) - .5 * jnp.dot(v, v) 
                -logprob_fn(x) - ldj + .5 * jnp.dot(v_, v_)
            )
        def kld_reverse(param, k, U, size):
            K = jax.random.split(k, size)
            return jnp.sum(jax.vmap(reverse_kld, (None, 0, 0))(param, U, K))
        def check_reverse(param, k, U, size):
            K = jax.random.split(k, size)
            return jnp.var(jax.vmap(lambda u, k: reverse_kld(param, u, k))(U, K))

        def forward_kld(param, x, k):
            v = jax.random.normal(k, (d,))
            u, v_, ldj = flow_inv(x, v, param, k)
            u = ravel_pytree(u)[0]
            return (
                logprob_fn(x) - .5 * jnp.dot(v, v) 
                + .5 * jnp.dot(u, u) - ldj + .5 * jnp.dot(v_, v_)
            )
        def kld_forward(param, k, X, size):
            K = jax.random.split(k, size)
            return jnp.sum(jax.vmap(forward_kld, (None, 0, 0))(param, X, K))
        def check_forward(param, k, X, size):
            K = jax.random.split(k, size)
            return jnp.var(jax.vmap(lambda x, k: forward_kld(param, x, k))(X, K))

        return (kld_reverse, check_reverse), (kld_forward, check_forward)


class renyi_alpha:
    def __new__(cls, logprob_fn, d, flow, flow_inv, alpha) -> Tuple:

        def reverse_renyi(param, u, k):
            v = jax.random.normal(k, (d,))
            x, v_, ldj = flow(u, v, param, k)
            u = ravel_pytree(u)[0]
            return (
                jnp.exp(logprob_fn(x) + ldj - .5 * jnp.dot(v_, v_)) 
                / jnp.exp(-.5 * jnp.dot(u, u) - .5 * jnp.dot(v, v))
            ) ** (1 - alpha)
        def renyi_reverse(param, k, U, size):
            K = jax.random.split(k, size)
            return jnp.log(jnp.sum(jax.vmap(
                lambda u, k: reverse_renyi(param, u, k) #** (1 - alpha)
            )(U, K))) / (1 - alpha)
        def check_reverse(param, k, U, size):
            K = jax.random.split(k, size)
            return jnp.var(jax.vmap(lambda u, k: reverse_renyi(param, u, k))(U, K))

        def forward_renyi(param, x, k):
            v = jax.random.normal(k, (d,))
            u, v_, ldj = flow_inv(x, v, param, k)
            u = ravel_pytree(u)[0]
            return (
                jnp.exp(-.5 * jnp.dot(u, u) + ldj - .5 * jnp.dot(v_, v_))
                / jnp.exp(logprob_fn(x) - .5 * jnp.dot(v, v))
            ) ** (1 - alpha)
        def renyi_forward(param, k, U, size):
            K = jax.random.split(k, size)
            return jnp.log(jnp.sum(jax.vmap(
                lambda u, k: forward_renyi(param, u, k) #** (1 - alpha)
            )(U, K))) / (1 - alpha)
        def check_forward(param, k, U, size):
            K = jax.random.split(k, size)
            return jnp.var(jax.vmap(lambda u, k: forward_renyi(param, u, k))(U, K))

        return (renyi_reverse, check_reverse), (renyi_forward, check_forward)