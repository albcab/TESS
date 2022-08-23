import abc
from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from jax.experimental.host_callback import id_print

import haiku as hk
import distrax as dx


def affine_coupling(params):
    return dx.ScalarAffine(shift=params[0], log_scale=params[1])

def rquad_spline_coupling(params):
    return dx.RationalQuadraticSpline(
        params, range_min=-6., range_max=6.)

        
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
    for _ in range(hidden_dims):
        layers.append(hk.Linear(d, w_init=hk.initializers.VarianceScaling(.01), b_init=hk.initializers.RandomNormal(.01)))
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


class Flow(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def flows(self):
        pass
    
    def get_utilities(self):
        forward_and_log_det = hk.transform(lambda u: self.flows().forward_and_log_det(u))
        inverse_and_log_det = hk.transform(lambda x: self.flows().inverse_and_log_det(x))

        def flow(u, param):
            u, unravel_fn = ravel_pytree(u)
            x, ldj = forward_and_log_det.apply(param, None, u)
            return unravel_fn(x), ldj
        
        def flow_inv(x, param):
            x, unravel_fn = ravel_pytree(x)
            u, ldj = inverse_and_log_det.apply(param, None, x)
            return unravel_fn(u), ldj

        return forward_and_log_det.init, flow, flow_inv


class Coupling(Flow):
    
    def __init__(self,
        # coupling_fn: Callable,
        d: int, n_flow: int,
        hidden_dims: int, non_linearity: Callable, norm: bool,
        num_bins: int = None,
    ):
        if num_bins:
            self.coupling_fn = rquad_spline_coupling
        else:
            self.coupling_fn = affine_coupling
        self.split = int(d/2 + .5)
        self.d = d
        self.n_flow = n_flow
        self.hidden_dims = hidden_dims
        self.non_linearity = non_linearity
        self.norm = norm
        self.num_bins = num_bins

    def flows(self):
        flows = []
        if self.num_bins:
            flows.append(shift_scale(self.d))
        for _ in range(self.n_flow):
            encoder = make_dense(self.split, self.hidden_dims, self.norm, self.non_linearity, self.num_bins)
            flows.append(dx.SplitCoupling(self.split, 1, encoder, self.coupling_fn, swap=True))
            decoder = make_dense(self.d - self.split, self.hidden_dims, self.norm, self.non_linearity, self.num_bins)
            flows.append(dx.SplitCoupling(self.split, 1, decoder, self.coupling_fn, swap=False))
        if self.num_bins:
            flows.append(shift_scale(self.d))
        return dx.Chain(flows)


class ShiftScale(Flow):

    def __init__(self, d):
        self.d = d

    def flows(self):
        return shift_scale(self.d)


def shift_scale(d):
    lin = hk.Sequential([
        hk.Linear(2 * d, w_init=jnp.zeros, b_init=hk.initializers.RandomNormal(.1)), 
        hk.Reshape((2, d), preserve_dims=-1)
    ])
    return dx.MaskedCoupling(jnp.zeros(d).astype(bool), lin, affine_coupling)
