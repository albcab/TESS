from typing import Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class kullback_liebler:
    def __new__(cls, logprob_fn, flow, flow_inv) -> Tuple:

        def reverse_kld(param, u):
            x, ldj = flow(u, param)
            u = ravel_pytree(u)[0]
            return (
                -.5 * jnp.dot(u, u)
                -logprob_fn(x) - ldj
            )
        def kld_reverse(param, U):
            return jnp.sum(jax.vmap(reverse_kld, (None, 0))(param, U))

        def forward_kld(param, x):
            u, ldj = flow_inv(x, param)
            u = ravel_pytree(u)[0]
            return (
                logprob_fn(x)
                + .5 * jnp.dot(u, u) - ldj
            )
        def kld_forward(param, X):
            return jnp.sum(jax.vmap(forward_kld, (None, 0))(param, X))

        return kld_reverse, kld_forward


class renyi_alpha:
    def __new__(cls, logprob_fn, flow, flow_inv, alpha) -> Tuple:

        def reverse_renyi(param, u):
            x, ldj = flow(u, param)
            u = ravel_pytree(u)[0]
            return jnp.exp(
                logprob_fn(x) + ldj
                +.5 * jnp.dot(u, u)
            ) ** (1 - alpha)
        def renyi_reverse(param, U):
            return jnp.log(jnp.sum(jax.vmap(
                lambda u: reverse_renyi(param, u) #** (1 - alpha)
            )(U))) / (1 - alpha)

        def forward_renyi(param, x):
            u, ldj = flow_inv(x, param)
            u = ravel_pytree(u)[0]
            return jnp.exp(
                -.5 * jnp.dot(u, u) + ldj
                -logprob_fn(x)
            ) ** (1 - alpha)
        def renyi_forward(param, U):
            return jnp.log(jnp.sum(jax.vmap(
                lambda u: forward_renyi(param, u) #** (1 - alpha)
            )(U))) / (1 - alpha)

        return renyi_reverse, renyi_forward