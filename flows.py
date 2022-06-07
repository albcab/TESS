from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.example_libraries import stax

from jax.experimental.host_callback import id_print

import haiku as hk
import distrax as dx

from nn_utils import affine_iaf_masks, MaskedLinear, Autoregressive

def make_iaf(d, n_hidden, non_linearity):
    masks = affine_iaf_masks(d, n_hidden)
    layers = []
    for mask in masks[:-1]:
        layers.append(MaskedLinear(mask))
        layers.append(non_linearity)
    layers.append(MaskedLinear(masks[-1]))
    layers.append(hk.Reshape((2, d), preserve_dims=-1))
    return hk.Sequential(layers)

def make_iaf_flow(d, n_flow, n_hidden, non_linearity, invert):
    def bijector_fn(params):
        return dx.ScalarAffine(shift=params[0], log_scale=params[1])
    P = jnp.flip(jnp.eye(d), 1)
    permute = lambda x: P @ x
    ldj_fn = lambda x: 0.

    flows = []
    for _ in range(n_flow):
        param_fn = make_iaf(d, n_hidden, non_linearity)
        flows.append(Autoregressive(d, param_fn, bijector_fn))
        if invert:
            flows.append(dx.Lambda(permute, permute, ldj_fn, ldj_fn, 1, 1, True))
    if invert:
        flows.append(dx.Lambda(permute, permute, ldj_fn, ldj_fn, 1, 1, True))
    return dx.Chain(flows)

        
def make_dense(d, hidden_dims, norm, non_linearity):
    layers = []
    for hd in hidden_dims:
        layers.append(hk.Linear(hd, w_init=hk.initializers.VarianceScaling(.1), b_init=hk.initializers.RandomNormal()))
        if norm:
            layers.append(hk.LayerNorm(-1, True, True))
        layers.append(non_linearity)
    layers.append(hk.Linear(2 * d, w_init=hk.initializers.VarianceScaling(.1), b_init=hk.initializers.RandomNormal()))
    layers.append(hk.Reshape((2, d), preserve_dims=-1))
    return hk.Sequential(layers)

def make_coupling_flow(d, n_flow, hidden_dims, non_linearity, norm):
    def bijector_fn(params):
        return dx.ScalarAffine(shift=params[0], log_scale=params[1])
    
    flows = []
    for _ in range(n_flow):
        encoder = make_dense(d, hidden_dims, norm, non_linearity)
        flows.append(dx.SplitCoupling(d, 1, encoder, bijector_fn, False))
        decoder = make_dense(d, hidden_dims, norm, non_linearity)
        flows.append(dx.SplitCoupling(d, 1, decoder, bijector_fn, True))
    return dx.Chain(flows)


class inverse_autoreg:
    def __new__(
        cls,
        logprob_fn: Callable,
        d: int, n_flow: int,
        hidden_dims: Sequence[int], non_linearity: Callable, invert: bool,
    ) -> Tuple:

        forward_and_log_det = hk.transform(lambda u: make_iaf_flow(d, n_flow, len(hidden_dims), non_linearity, invert).forward_and_log_det(u))
        inverse_and_log_det = hk.transform(lambda x: make_iaf_flow(d, n_flow, len(hidden_dims), non_linearity, invert).inverse_and_log_det(x))

        def flow(u, v, param, rng=None):
            u, unravel_fn = ravel_pytree(u)
            x, ldj = forward_and_log_det.apply(param, rng, u)
            return unravel_fn(x), v, ldj
        
        def flow_inv(x, v, param, rng=None):
            x, unravel_fn = ravel_pytree(x)
            u, ldj = inverse_and_log_det.apply(param, rng, x)
            return unravel_fn(u), v, ldj

        reverse_kld, forward_kld = kullback_liebler(logprob_fn, d, flow, flow_inv)

        return forward_and_log_det.init, flow, flow_inv, reverse_kld, forward_kld


class coupling_dense:
    def __new__(
        cls,
        logprob_fn: Callable,
        d: int, n_flow: int,
        hidden_dims: Sequence[int], non_linearity: Callable, norm: bool,
    ) -> Tuple:

        forward_and_log_det = hk.transform(lambda uv: make_coupling_flow(d, n_flow, hidden_dims, non_linearity, norm).forward_and_log_det(uv))
        inverse_and_log_det = hk.transform(lambda xv: make_coupling_flow(d, n_flow, hidden_dims, non_linearity, norm).inverse_and_log_det(xv))

        def param_init(rng, p):
            return forward_and_log_det.init(rng, jnp.concatenate([p, p]))

        def flow(u, v, param, rng=None):
            u, unravel_fn = ravel_pytree(u)
            xv, ldj = forward_and_log_det.apply(param, rng, jnp.concatenate([u, v]))
            return unravel_fn(xv.at[:d].get()), xv.at[-d:].get(), ldj
        
        def flow_inv(x, v, param, rng=None):
            x, unravel_fn = ravel_pytree(x)
            uv, ldj = inverse_and_log_det.apply(param, rng, jnp.concatenate([x, v]))
            return unravel_fn(uv.at[:d].get()), uv.at[-d:].get(), ldj

        reverse_kld, forward_kld = kullback_liebler(logprob_fn, d, flow, flow_inv)

        return param_init, flow, flow_inv, reverse_kld, forward_kld


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

        def forward_kld(param, x, k):
            v = jax.random.normal(k, (d,))
            u, v_, ldj = flow_inv(x, v, param, k)
            u = ravel_pytree(u)[0]
            return (
                logprob_fn(x) - .5 * jnp.dot(v, v) 
                + .5 * jnp.dot(u, u) - ldj + .5 * jnp.dot(v_, v_)
            )

        return reverse_kld, forward_kld

# class renyi_alpha:
#     def __new__(cls, logprob_fn, d, flow, flow_inv) -> Tuple:

#         def 