from typing import Callable, Dict, NamedTuple, Sequence, Tuple, Union, Iterable, Mapping, Any

import jax._src.prng as prng
import numpy as np

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.example_libraries import stax

from jax.experimental.host_callback import id_print

from mcmc_utils import inference_loop
from nn_utils import optimize, affine_iaf_masks, MaskedDense, Dropout

import jaxopt
from optax import GradientTransformation

Array = Union[np.ndarray, jnp.ndarray]

PyTree = Union[Array, Iterable[Array], Mapping[Any, Array]]

PRNGKey = prng.PRNGKeyArray

Optim = Union[GradientTransformation, jaxopt.OptaxSolver, jaxopt.LBFGS]


class SamplingAlgorithm(NamedTuple):
    init: Callable
    step: Callable

class SliceState(NamedTuple):
    position: PyTree
    logprob: PyTree

class SliceInfo(NamedTuple):
    momentum: PyTree
    theta: float
    subiter: int


class elliptical_slice:
    def __new__(
        cls,
        logprob_fn: Callable, d: int,
    ) -> SamplingAlgorithm:
        
        def init_fn(position: PyTree):
            return SliceState(position, 0)

        def slice_fn(p, m):
            return logprob_fn(p) -.5 * jnp.dot(m, m)
        def momentum_generator(rng_key, position):
            return jax.random.normal(rng_key, shape=(d,))
        def step_fn(rng_key: PRNGKey, state: SliceState):
            proposal_generator = tess_proposal(
                slice_fn, momentum_generator, 
                lambda u, v: (u, v), lambda x, v: (x, v)
            )
            return proposal_generator(rng_key, state)

        return SamplingAlgorithm(init_fn, step_fn)


class atransp_elliptical_slice:
    def __new__(
        cls,
        logprob_fn: Callable,
        optim: Optim, d: int, n_flow: int,
        hidden_dims: Sequence[int], non_linearity: Callable = stax.Tanh, norm: bool = False
    ) -> SamplingAlgorithm:
        
        layers = []
        for hd in hidden_dims:
            layers.append(stax.Dense(hd))
            if norm:
                layers.append(stax.BatchNorm(axis=0))
            # layers.append(Dropout(1/3))
            layers.append(non_linearity)
        layers.append(stax.Dense(2 * d))
        param_init_, Psi_ = stax.serial(*layers)
        def param_init(key, shape):
            keys = jax.random.split(key, n_flow)
            def enc_dec(k):
                ke, kd = jax.random.split(k)
                enc_param = jax.tree_map(lambda x: x * 1., param_init_(ke, shape)[1])
                dec_param = jax.tree_map(lambda x: x * 1., param_init_(kd, shape)[1])
                return {'enc': enc_param, 'dec': dec_param}
            return None, jax.vmap(enc_dec)(keys)
        Psi = lambda param, v, rng, mode: Psi_(param, v, rng=rng, mode=mode).reshape(2, -1)

        def T(u, v, param, rng=jax.random.PRNGKey(0), mode='sample'):
            u, unravel_fn = ravel_pytree(u)
            def flow_iter(carry, param_rng):
                x, v, ldj = carry
                param, rng = param_rng
                ke, kd = jax.random.split(rng)
                psi_v = Psi(param['enc'], x, ke, mode)
                v = jnp.exp(psi_v[1]) * v + psi_v[0]
                ldj += jnp.sum(psi_v[1])
                psi = Psi(param['dec'], v, kd, mode)
                x = jnp.exp(psi[1]) * x + psi[0]
                ldj += jnp.sum(psi[1])
                return (x, v, ldj), None
            rngs = jax.random.split(rng, n_flow)
            (x, v, ldj), _ = jax.lax.scan(flow_iter, (u, v, 0.), (param, rngs))
            return unravel_fn(x), v, ldj
        def T_inv(x, v, param, rng=jax.random.PRNGKey(0), mode='sample'):
            x, unravel_fn = ravel_pytree(x)
            rev_param = jax.tree_util.tree_map(lambda x: x[::-1], param)
            rev_rng = jnp.flip(jax.random.split(rng, n_flow), axis=0)
            def flow_iter(carry, param_rng):
                u, v, nldj = carry
                param, rng = param_rng
                ke, kd = jax.random.split(rng)
                psi = Psi(param['dec'], v, kd, mode)
                u = (u - psi[0]) / jnp.exp(psi[1])
                nldj += jnp.sum(psi[1])
                psi_v = Psi(param['enc'], u, ke, mode)
                v = (v - psi_v[0]) / jnp.exp(psi_v[1])
                nldj += jnp.sum(psi_v[1])
                return (u, v, nldj), None
            (u, v, nldj), _ = jax.lax.scan(flow_iter, (x, v, 0.), (rev_param, rev_rng))
            return unravel_fn(u), v, nldj

        def pi_tilde(param, u, k):
            v = jax.random.normal(k, (d,))
            x, v, ldj = T(u, v, param, k, mode='train')
            return -logprob_fn(x) + .5 * jnp.dot(v, v) - ldj

        def P(param, u, k):
            v = jax.random.normal(k, (d,))
            lp_v = -.5 * jnp.dot(v, v) 
            x, v, ldj = T(u, v, param, k, mode='train')
            u, _ = ravel_pytree(u)
            return -.5 * jnp.dot(u, u) + lp_v - logprob_fn(x) + .5 * jnp.dot(v, v) - ldj

        def init_fn(
            rng: PRNGKey, position: PyTree, 
            batch_size: int = 1000, batch_iter: int = 5, tol: float = 1e-0, maxiter : int = 1e6,
        ):
            n_batch = int(batch_size / batch_iter)
            def kld0(param, k, U):
                K = jax.random.split(k, n_batch)
                return jnp.sum(jax.vmap(pi_tilde, (None, 0, 0))(param, U, K))
            def check0(param, k, U):
                K = jax.vmap(lambda ki: jax.random.split(ki, n_batch))(jax.random.split(k, batch_iter))
                return jnp.var(jax.vmap(jax.vmap(lambda u, k: P(param, u, k)))(U, K))

            _, unraveler_fn = ravel_pytree(position)
            ku, ko = jax.random.split(rng)
            U = jax.vmap(jax.vmap(unraveler_fn))(jax.random.normal(ku, shape=(batch_iter, n_batch, d)))
            param = param_init(ko, (d,))[1]
            param, err = optimize(param, kld0, check0, optim, tol, maxiter, ko, U)
            return SliceState(position, 0), param, err

        def slice_fn(p, m):
            return logprob_fn(p) -.5 * jnp.dot(m, m)
        def momentum_generator(rng_key, position):
            return jax.random.normal(rng_key, shape=(d,))
        def step_fn(rng_key: PRNGKey, state: SliceState, param: PyTree):
            proposal_generator = tess_proposal(
                slice_fn, momentum_generator, 
                lambda u, v: T(u, v, param)[:-1], lambda x, v: T_inv(x, v, param)[:-1]
            )
            return proposal_generator(rng_key, state)

        def phi_tilde(param, x, k):
            v = jax.random.normal(k, (d,))
            u, v, nldj = T_inv(x, v, param, k, mode='train')
            u = ravel_pytree(u)[0]
            return .5 * jnp.dot(u, u) + .5 * jnp.dot(v, v) + nldj

        def Z(param, x, k):
            v = jax.random.normal(k, (d,))
            lp_v = -.5 * jnp.dot(v, v)
            u, v, nldj = T_inv(x, v, param, k, mode='train')
            u = ravel_pytree(u)[0]
            return logprob_fn(x) + lp_v + .5 * jnp.dot(u, u) + .5 * jnp.dot(v, v) + nldj
        
        def warm_fn(
            rng_key: PRNGKey, state: SliceState, param: PyTree, 
            n_epoch: int = 10, batch_size: int = 1000, batch_iter: int = 10, 
            tol: float = 1e-0, maxiter: int = 1e6,
        ):
            n_batch = int(batch_size / batch_iter)
            def kld_warm(param, k, X):
                K = jax.random.split(k, n_batch)
                return jnp.sum(jax.vmap(phi_tilde, (None, 0, 0))(param, X, K))
            def check(param, k, X):
                K = jax.vmap(lambda ki: jax.random.split(ki, n_batch))(jax.random.split(k, batch_iter))
                return jnp.var(jax.vmap(jax.vmap(lambda x, k: Z(param, x, k)))(X, K))

            rng_key, ks, kc = jax.random.split(rng_key, 3)
            states, info = inference_loop(ks, state, step_fn, batch_size, param)
            X = jax.tree_map(lambda x: jax.random.choice(kc, x, (batch_iter, n_batch)), states.position)

            def one_epoch(carry, key):
                state, param, X = carry
                param, err = optimize(param, kld_warm, check, optim, tol, maxiter, key, X)
                ks, kc = jax.random.split(key)
                states, info = inference_loop(ks, state, step_fn, batch_size, param)
                X = jax.tree_map(lambda x, y: jax.random.choice(kc, jnp.concatenate([x, *y]), (batch_iter, n_batch)), states.position, X)
                return (state, param, X), err
            rng_keys = jax.random.split(rng_key, n_epoch)
            (state, param, X), err = jax.lax.scan(one_epoch, (state, param, X), rng_keys[:-1])
            param, err_ = optimize(param, kld_warm, check, optim, tol, maxiter, rng_keys[-1], X)
            return (state, param), jnp.hstack([err, err_])
        
        return SamplingAlgorithm(init_fn, step_fn), warm_fn


class neutra:
    def __new__(
        cls,
        logprob_fn: Callable,
        optim: Optim, d: int, n_flow: int,
        n_hidden: Sequence[int], non_linearity: Callable = stax.Elu,
    ) -> SamplingAlgorithm:

        masks = affine_iaf_masks(d, len(n_hidden))
        layers = []
        for mask in masks[:-1]:
            layers.append(MaskedDense(mask))
            layers.append(non_linearity)
        layers.append(MaskedDense(masks[-1]))
        param_init_, Psi_ = stax.serial(*layers)
        def param_init(key, shape):
            keys = jax.random.split(key, n_flow)
            return None, jax.vmap(lambda k: param_init_(k, shape)[1])(keys)
        Psi = lambda param, u: Psi_(param, u).reshape(2, -1)

        def T(u, param):
            u, unravel_fn = ravel_pytree(u)
            def flow_iter(carry, param):
                u, ldj = carry
                psi = Psi(param, u)
                x = jnp.exp(psi[1]) * u + psi[0]
                x = jax.tree_util.tree_map(lambda x: x[::-1], x)
                ldj += jnp.sum(psi[1])
                return (x, ldj), None
            (x, ldj), _ = jax.lax.scan(flow_iter, (u, 0.), param)
            x = jax.tree_util.tree_map(lambda x: x[::-1], x)
            return unravel_fn(x), ldj

        def pi_tilde(param, u):
            x, ldj = T(u, param)
            return -logprob_fn(x) - ldj
        kld = lambda param, k, U: jnp.sum(jax.vmap(pi_tilde, (None, 0))(param, U))

        def P(param, u, k):
            x, ldj = T(u, param)
            u, _ = ravel_pytree(u)
            return -.5 * jnp.dot(u, u) - logprob_fn(x) - ldj
        check = lambda param, k, U: jnp.var(jax.vmap(jax.vmap(lambda u: P(param, u, k)))(U))

        def init_fn(
            rng: PRNGKey, position: PyTree, 
            batch_size: int = 1000, batch_iter: int = 5, tol: float = 1e-0, maxiter : int = 1e6,
        ):
            _, unraveler_fn = ravel_pytree(position)
            ku, kp = jax.random.split(rng)

            n_batch = int(batch_size / batch_iter)
            U = jax.vmap(jax.vmap(unraveler_fn))(jax.random.normal(ku, shape=(batch_iter, n_batch, d)))

            init_param = param_init(kp, (d,))[1]
            param, err = optimize(init_param, kld, check, optim, tol, maxiter, kp, U)
            pullback_fn = lambda u: -pi_tilde(param, u)
            return pullback_fn, param, err

        return init_fn


def ellipsis2(p, m, theta, mu=0.):
    x, unraveler = ravel_pytree(p)
    return (
        unraveler((x - mu) * jnp.cos(theta) + (m - mu) * jnp.sin(theta) + mu),
        (m - mu) * jnp.cos(theta) - (x - mu) * jnp.sin(theta) + mu
    )

def tess_proposal(
    slice_fn: Callable, 
    momentum_generator: Callable,
    T: Callable, T_inv: Callable,
    # Psi: Callable,
) -> Callable:

    def generate(rng_key: PRNGKey, state: SliceState) -> Tuple[SliceState, SliceInfo]:
        position, _ = state
        # id_print(position)
        kmomentum, kunif, ktheta = jax.random.split(rng_key, 3)
        # step 1: sample momentum
        momentum = momentum_generator(kmomentum, position)
        # id_print(momentum)
        #step 2-3: get slice (y)
        logy = slice_fn(position, momentum) + jnp.log(jax.random.uniform(kunif))
        #step 4: get u
        # u_position = T_inv(position, Psi(momentum))
        u_position, momentum = T_inv(position, momentum)
        # id_print(u_position)
        # id_print(momentum)
        #step 5-6: get theta (ellipsis move), set inital interval
        theta = 2 * jnp.pi * jax.random.uniform(ktheta)
        theta_min = theta - 2 * jnp.pi
        theta_max = theta
        #step 7-8: proposal
        u, m = ellipsis2(u_position, momentum, theta)
        # id_print(u)
        # id_print(m)
        #step 9: get new position
        # p = T(u, Psi(m))
        p, m = T(u, m)
        # id_print(p)
        # id_print(m)
        #step 10-20: acceptance
        slice = slice_fn(p, m)

        def while_fun(vals):
            rng, _, subiter, theta, theta_min, theta_max, *_ = vals
            rng, thetak = jax.random.split(rng)
            theta = jax.random.uniform(thetak, minval=theta_min, maxval=theta_max)
            u, m = ellipsis2(u_position, momentum, theta)
            # p = T(u, Psi(m))
            p, m = T(u, m)
            slice = slice_fn(p, m)
            theta_min = jnp.where(theta < 0, theta, theta_min)
            theta_max = jnp.where(theta > 0, theta, theta_max)
            subiter += 1
            return rng, slice, subiter, theta, theta_min, theta_max, p, m

        _, slice, subiter, theta, *_, position, momentum = jax.lax.while_loop(
            lambda vals: vals[1] <= logy, 
            while_fun, 
            (rng_key, slice, 1, theta, theta_min, theta_max, p, m)
        )
        # id_print(position)
        return (SliceState(position, slice), 
            SliceInfo(momentum, theta, subiter))

    return generate


def ellipsis(p, m, theta, mu=0.):
    x, unraveler = ravel_pytree(p)
    v = ravel_pytree(m)[0]
    return (
        unraveler((x - mu) * jnp.cos(theta) + (v - mu) * jnp.sin(theta) + mu),
        unraveler((v - mu) * jnp.cos(theta) - (x - mu) * jnp.sin(theta) + mu)
    )

def elliptical_proposal(
    loglikelihood_fn: Callable, 
    momentum_generator: Callable
) -> Callable:

    def generate(rng_key: PRNGKey, state: SliceState) -> Tuple[SliceState, SliceInfo]:
        position, loglikelihood = state
        kmomentum, kunif, ktheta = jax.random.split(rng_key, 3)
        # step 1: sample momentum
        momentum = momentum_generator(kmomentum, position)
        # #step 2: get slice (y)
        logy = loglikelihood + jnp.log(jax.random.uniform(kunif))
        #step 3: get theta (ellipsis move), set inital interval
        theta = 2 * jnp.pi * jax.random.uniform(ktheta)
        theta_min = theta - 2 * jnp.pi
        theta_max = theta
        #step 4: proposal
        p, m = ellipsis(position, momentum, theta)
        #step 5: acceptance
        loglikelihood = loglikelihood_fn(p)

        def while_fun(vals):
            rng, _, subiter, theta, theta_min, theta_max, *_ = vals
            rng, thetak = jax.random.split(rng)
            theta = jax.random.uniform(thetak, minval=theta_min, maxval=theta_max)
            p, m = ellipsis(position, momentum, theta)
            loglikelihood = loglikelihood_fn(p)
            theta_min = jnp.where(theta < 0, theta, theta_min)
            theta_max = jnp.where(theta > 0, theta, theta_max)
            subiter += 1
            return rng, loglikelihood, subiter, theta, theta_min, theta_max, p, m

        _, loglikelihood, subiter, theta, *_, position, momentum = jax.lax.while_loop(
            lambda vals: vals[1] <= logy, 
            while_fun, 
            (rng_key, loglikelihood, 1, theta, theta_min, theta_max, p, m)
        )
        return (SliceState(position, loglikelihood), 
            SliceInfo(momentum, theta, subiter))

    return generate
