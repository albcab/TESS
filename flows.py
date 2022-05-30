from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.example_libraries import stax

from jax.experimental.host_callback import id_print

import haiku as hk
import distrax as dx

        
def make_dense(d, hidden_dims, norm, non_linearity):
    layers = []
    for hd in hidden_dims:
        layers.append(hk.Linear(hd))
        if norm:
            layers.append(hk.BatchNorm(True, True, .9))
        layers.append(non_linearity)
    layers.append(hk.Linear(2 * d))
    layers.append(hk.Reshape((d, 2), preserve_dims=-1))
    return hk.Sequential(layers)

def make_coupling_flow(d, n_flow, hidden_dims, non_linearity, norm):
    def bijector_fn(params):
        return dx.ScalarAffine(shift=params[..., 0], log_scale=params[..., 1])
    
    layers = []
    for _ in range(n_flow):
        encoder = make_dense(d, hidden_dims, norm, non_linearity)
        layers.append(dx.SplitCoupling(d, 1, encoder, bijector_fn, False))
        decoder = make_dense(d, hidden_dims, norm, non_linearity)
        layers.append(dx.SplitCoupling(d, 1, decoder, bijector_fn, True))
    return dx.Chain(layers)

class coupling_dense:
    def __new__(
        cls,
        logprob_fn: Callable,
        d: int, n_flow: int,
        hidden_dims: Sequence[int], non_linearity: Callable, norm: bool = False
    ) -> Tuple:

        forward_and_log_det = hk.transform(lambda uv: make_coupling_flow(d, n_flow, hidden_dims, non_linearity, norm).forward_and_log_det(uv))
        inverse_and_log_det = hk.transform(lambda xv: make_coupling_flow(d, n_flow, hidden_dims, non_linearity, norm).inverse_and_log_det(xv))

        def flow(u, v, param, rng=None):
            u, unravel_fn = ravel_pytree(u)
            xv, ldj = forward_and_log_det.apply(param, rng, jnp.concatenate([u, v]))
            return unravel_fn(xv.at[:d].get()), xv.at[-d:].get(), ldj
        
        def flow_inv(x, v, param, rng=None):
            x, unravel_fn = ravel_pytree(x)
            uv, ldj = inverse_and_log_det.apply(param, rng, jnp.concatenate([x, v]))
            return unravel_fn(uv.at[:d].get()), uv.at[-d:].get(), ldj

        def reverse_kld(param, u, k):
            v = jax.random.normal(k, (d,))
            x, v_, ldj = flow(u, v, param, k)
            u = ravel_pytree(u)[0]
            return (
                -.5 * jnp.dot(u, u) - .5 * jnp.dot(v, v) 
                -logprob_fn(x) - ldj + .5 * jnp.dot(v_, v_)
            )

        def forward_kld(param, x, k):
            v = jax.random.normal(k, (d,))
            u, v_, ldj = flow_inv(x, v, param, k)
            u = ravel_pytree(u)[0]
            return (
                logprob_fn(x) - .5 * jnp.dot(v, v) 
                + .5 * jnp.dot(u, u) - ldj + .5 * jnp.dot(v_, v_)
            )

        return forward_and_log_det.init, flow, flow_inv, reverse_kld, forward_kld


# class coupling_dense:
#     def __new__(
#         cls,
#         logprob_fn: Callable,
#         d: int, n_flow: int,
#         hidden_dims: Sequence[int], non_linearity: Callable, norm: bool = False
#     ) -> Tuple:

#         layers = []
#         for hd in hidden_dims:
#             layers.append(stax.Dense(hd))
#             if norm:
#                 layers.append(stax.BatchNorm(axis=0))
#             # layers.append(Dropout(1/3))
#             layers.append(non_linearity)
#         layers.append(stax.Dense(2 * d))
#         param_init_, Psi_ = stax.serial(*layers)
#         def param_init(key, shape):
#             keys = jax.random.split(key, n_flow)
#             def enc_dec(k):
#                 ke, kd = jax.random.split(k)
#                 enc_param = jax.tree_map(lambda x: x * 1., param_init_(ke, shape)[1])
#                 dec_param = jax.tree_map(lambda x: x * 1., param_init_(kd, shape)[1])
#                 return {'enc': enc_param, 'dec': dec_param}
#             return None, jax.vmap(enc_dec)(keys)
#         Psi = lambda param, v, rng, mode: Psi_(param, v, rng=rng, mode=mode).reshape(2, -1)

#         def T(u, v, param, rng=jax.random.PRNGKey(0), mode='sample'):
#             u, unravel_fn = ravel_pytree(u)
#             def flow_iter(carry, param_rng):
#                 x, v, ldj = carry
#                 param, rng = param_rng
#                 ke, kd = jax.random.split(rng)
#                 psi_v = Psi(param['enc'], x, ke, mode)
#                 v = jnp.exp(psi_v[1]) * v + psi_v[0]
#                 ldj += jnp.sum(psi_v[1])
#                 psi = Psi(param['dec'], v, kd, mode)
#                 x = jnp.exp(psi[1]) * x + psi[0]
#                 ldj += jnp.sum(psi[1])
#                 return (x, v, ldj), None
#             rngs = jax.random.split(rng, n_flow)
#             (x, v, ldj), _ = jax.lax.scan(flow_iter, (u, v, 0.), (param, rngs))
#             return unravel_fn(x), v, ldj
        
#         def T_inv(x, v, param, rng=jax.random.PRNGKey(0), mode='sample'):
#             x, unravel_fn = ravel_pytree(x)
#             rev_param = jax.tree_util.tree_map(lambda x: x[::-1], param)
#             rev_rng = jnp.flip(jax.random.split(rng, n_flow), axis=0)
#             def flow_iter(carry, param_rng):
#                 u, v, nldj = carry
#                 param, rng = param_rng
#                 ke, kd = jax.random.split(rng)
#                 psi = Psi(param['dec'], v, kd, mode)
#                 u = (u - psi[0]) / jnp.exp(psi[1])
#                 nldj += jnp.sum(psi[1])
#                 psi_v = Psi(param['enc'], u, ke, mode)
#                 v = (v - psi_v[0]) / jnp.exp(psi_v[1])
#                 nldj += jnp.sum(psi_v[1])
#                 return (u, v, nldj), None
#             (u, v, nldj), _ = jax.lax.scan(flow_iter, (x, v, 0.), (rev_param, rev_rng))
#             return unravel_fn(u), v, nldj

#         def pi_tilde(param, u, k):
#             v = jax.random.normal(k, (d,))
#             x, v, ldj = T(u, v, param, k, mode='train')
#             return -logprob_fn(x) + .5 * jnp.dot(v, v) - ldj

#         def P(param, u, k):
#             v = jax.random.normal(k, (d,))
#             lp_v = -.5 * jnp.dot(v, v) 
#             x, v, ldj = T(u, v, param, k, mode='train')
#             u, _ = ravel_pytree(u)
#             return -.5 * jnp.dot(u, u) + lp_v - logprob_fn(x) + .5 * jnp.dot(v, v) - ldj

#         def phi_tilde(param, x, k):
#             v = jax.random.normal(k, (d,))
#             u, v, nldj = T_inv(x, v, param, k, mode='train')
#             u = ravel_pytree(u)[0]
#             return .5 * jnp.dot(u, u) + .5 * jnp.dot(v, v) + nldj

#         def Z(param, x, k):
#             v = jax.random.normal(k, (d,))
#             lp_v = -.5 * jnp.dot(v, v)
#             u, v, nldj = T_inv(x, v, param, k, mode='train')
#             u = ravel_pytree(u)[0]
#             return logprob_fn(x) + lp_v + .5 * jnp.dot(u, u) + .5 * jnp.dot(v, v) + nldj

#         return param_init, T, T_inv, pi_tilde, P, phi_tilde, Z