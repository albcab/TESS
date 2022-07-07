from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from jax.experimental.host_callback import id_print

import haiku as hk
import distrax as dx

from nn_utils import affine_iaf_masks, MaskedLinear, Autoregressive


def affine_coupling(params):
    return dx.ScalarAffine(shift=params[0], log_scale=params[1])

def rquad_spline_coupling(params):
    return dx.RationalQuadraticSpline(
        params, range_min=-4., range_max=4.)


def make_iaf(d, n_hidden, non_linearity):
    masks = affine_iaf_masks(d, n_hidden)
    layers = []
    for mask in masks[:-1]:
        layers.append(MaskedLinear(mask, 
            w_init=hk.initializers.VarianceScaling(.01), b_init=hk.initializers.RandomNormal(.01)))
        layers.append(non_linearity)
    layers.append(MaskedLinear(masks[-1], 
        w_init=hk.initializers.VarianceScaling(.01), b_init=hk.initializers.RandomNormal(.01)))
    layers.append(hk.Reshape((2, d), preserve_dims=-1))
    return hk.Sequential(layers)

        
def make_dense(d, hidden_dims, norm, non_linearity, num_bins):
    # layers = [hk.nets.MLP(hidden_dims, 
    #     w_init=hk.initializers.VarianceScaling(.01), b_init=hk.initializers.RandomNormal(.01), 
    #     activation=non_linearity, activate_final=True)
    # ]
    # if num_bins:
    #     layers.append(
    #         hk.Linear(3 * num_bins + 1, w_init=hk.initializers.VarianceScaling(.01), b_init=hk.initializers.RandomNormal(.01))
    #     )
    # else:
    #     layers.extend([
    #         hk.Linear(2 * d, w_init=hk.initializers.VarianceScaling(.01), b_init=hk.initializers.RandomNormal(.01)),
    #         hk.Reshape((2, d), preserve_dims=-1)
    #     ])
    # return hk.Sequential(layers)
    layers = []
    for hd in hidden_dims:
        layers.append(hk.Linear(hd, w_init=hk.initializers.VarianceScaling(.01), b_init=hk.initializers.RandomNormal(.01)))
        if norm:
            layers.append(hk.LayerNorm(-1, True, True))
        layers.append(non_linearity)
    if num_bins:
        layers.append(
            hk.Linear(3 * num_bins + 1, w_init=hk.initializers.VarianceScaling(.01), b_init=hk.initializers.RandomNormal(.01))
        )
    else:
        layers.extend([
            hk.Linear(2 * d, w_init=hk.initializers.VarianceScaling(.01), b_init=hk.initializers.RandomNormal(.01)),
            hk.Reshape((2, d), preserve_dims=-1)
        ])
    return hk.Sequential(layers)


class coupling_auto:
    def __new__(cls,
        # coupling_fn: Callable,
        d: int, n_flow: int,
        hidden_dims: Sequence[int], non_linearity: Callable, norm: bool,
        num_bins: int = None,
    ) -> Tuple:
        
        if num_bins:
            coupling_fn = rquad_spline_coupling
        else:
            coupling_fn = affine_coupling

        split = int(d/2 + .5)
        def flows():
            flows = []
            if num_bins:
                lin = hk.Sequential([hk.Linear(2 * d, w_init=jnp.zeros, b_init=hk.initializers.RandomNormal(.1)), hk.Reshape((2, d), preserve_dims=-1)])
                flows.append(dx.MaskedCoupling(jnp.zeros(d).astype(bool), lin, affine_coupling))
            for _ in range(n_flow):
                encoder = make_dense(split, hidden_dims, norm, non_linearity, num_bins)
                flows.append(dx.SplitCoupling(split, 1, encoder, coupling_fn, swap=True))
                decoder = make_dense(d - split, hidden_dims, norm, non_linearity, num_bins)
                flows.append(dx.SplitCoupling(split, 1, decoder, coupling_fn, swap=False))
            if num_bins:
                lin = hk.Sequential([hk.Linear(2 * d, w_init=jnp.zeros, b_init=hk.initializers.RandomNormal(.1)), hk.Reshape((2, d), preserve_dims=-1)])
                flows.append(dx.MaskedCoupling(jnp.zeros(d).astype(bool), lin, affine_coupling))
            return dx.Chain(flows)

        forward_and_log_det = hk.transform(lambda u: flows().forward_and_log_det(u))
        inverse_and_log_det = hk.transform(lambda x: flows().inverse_and_log_det(x))

        def flow(u, v, param, rng=None):
            u, unravel_fn = ravel_pytree(u)
            x, ldj = forward_and_log_det.apply(param, rng, u)
            return unravel_fn(x), v, ldj
        
        def flow_inv(x, v, param, rng=None):
            x, unravel_fn = ravel_pytree(x)
            u, ldj = inverse_and_log_det.apply(param, rng, x)
            return unravel_fn(u), v, ldj

        return forward_and_log_det.init, flow, flow_inv


class inverse_autoreg:
    def __new__(cls,
        # coupling_fn: Callable, 
        d: int, n_flow: int,
        hidden_dims: Sequence[int], non_linearity: Callable, invert: bool,
    ) -> Tuple:

        print("BROKEN")

        coupling_fn = affine_coupling

        n_hidden = len(hidden_dims)
        P = jnp.flip(jnp.eye(d), 1)
        permute = lambda x: P @ x
        ldj_fn = lambda x: 0.
        def flows():
            flows = []
            for _ in range(n_flow):
                conditioner = make_iaf(d, n_hidden, non_linearity)
                flows.append(Autoregressive(d, conditioner, coupling_fn))
                if invert:
                    flows.append(dx.Lambda(permute, permute, ldj_fn, ldj_fn, 1, 1, True))
            if invert:
                flows.append(dx.Lambda(permute, permute, ldj_fn, ldj_fn, 1, 1, True))
            return dx.Chain(flows)

        forward_and_log_det = hk.transform(lambda u: flows().forward_and_log_det(u))
        inverse_and_log_det = hk.transform(lambda x: flows().inverse_and_log_det(x))

        def flow(u, v, param, rng=None):
            u, unravel_fn = ravel_pytree(u)
            x, ldj = forward_and_log_det.apply(param, rng, u)
            return unravel_fn(x), v, ldj
        
        def flow_inv(x, v, param, rng=None):
            x, unravel_fn = ravel_pytree(x)
            u, ldj = inverse_and_log_det.apply(param, rng, x)
            return unravel_fn(u), v, ldj

        return forward_and_log_det.init, flow, flow_inv


class coupling_latent:
    def __new__(cls,
        # coupling_fn: Callable, 
        d: int, n_flow: int,
        hidden_dims: Sequence[int], non_linearity: Callable, norm: bool,
        num_bins: int = None,
    ) -> Tuple:

        if num_bins:
            coupling_fn = rquad_spline_coupling
        else:
            coupling_fn = affine_coupling
        
        def flows():
            flows = []
            for _ in range(n_flow):
                encoder = make_dense(d, hidden_dims, norm, non_linearity, num_bins)
                flows.append(dx.SplitCoupling(d, 1, encoder, coupling_fn, True))
                decoder = make_dense(d, hidden_dims, norm, non_linearity, num_bins)
                flows.append(dx.SplitCoupling(d, 1, decoder, coupling_fn, False))
            # if num_bins:
            #     flows.append() something that scales the transformation to the scale and bias of target
            return dx.Chain(flows)

        forward_and_log_det = hk.transform(lambda uv: flows().forward_and_log_det(uv))
        inverse_and_log_det = hk.transform(lambda xv: flows().inverse_and_log_det(xv))

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

        return param_init, flow, flow_inv
