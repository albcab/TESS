from typing import Callable, Dict, NamedTuple, Sequence, Tuple, Union, Iterable, Mapping, Any

import jax._src.prng as prng
import numpy as np

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

from jax.experimental.host_callback import id_print

from mcmc_utils import inference_loop
from nn_utils import optimize, optimize2

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
        optim: Optim, d, param_init, flow, flow_inv, reverse, forward
    ) -> SamplingAlgorithm:

        def init_fn(
            rng: PRNGKey, position: PyTree, 
            batch_size: int, batch_iter: int, tol: float, maxiter: int,
        ):
            p, unraveler_fn = ravel_pytree(position)
            ku, ko = jax.random.split(rng)
            U = jax.vmap(unraveler_fn)(jax.random.normal(ku, shape=(batch_iter * batch_size, d)))

            param = param_init(ko, p)
            param, err = optimize(param, reverse, optim, tol, maxiter, ko, U, batch_iter, batch_size)

            # #starting from flow observation
            # ku, kv = jax.random.split(ko)
            # u = unraveler_fn(jax.random.normal(ku, (d,)))
            # v = jax.random.normal(kv, (d,))
            # position, *_ = flow(u, v, param)

            return SliceState(position, 0), param, err

        def momentum_generator(rng_key, position):
            return jax.random.normal(rng_key, shape=(d,))
        def step_fn(rng_key: PRNGKey, state: SliceState, param: PyTree):
            def slice_fn(u, m):
                x, v, ldj = flow(u, m, param)
                return logprob_fn(x) + ldj - .5 * jnp.dot(v, v)
            proposal_generator = tess_proposal(
                slice_fn, momentum_generator, 
                lambda u, v: flow(u, v, param)[:-1], lambda x, v: flow_inv(x, v, param)[:-1]
            )
            return proposal_generator(rng_key, state)
        
        def warm_fn(
            rng_key: PRNGKey, state: SliceState, param: PyTree, 
            n_epoch: int, batch_size: int, batch_iter: int, 
            tol: float, maxiter: int,
        ):
            rng_key, ks, kc = jax.random.split(rng_key, 3)
            states, info = inference_loop(ks, state, step_fn, batch_size * batch_iter, param)
            X = states.position

            # #Different optimization path
            # def build_check(i):
            #     def check(param, k, X):
            #         K = jax.random.split(k, batch_size * (i+1))
            #         return jnp.var(jax.vmap(lambda x, k: forward_kld(param, x, k))(X, K))
            #     return check
            # list_params = []
            # for i in range(n_epoch):
            #     param, var = optimize2(param, kld_warm, build_check(i), optim, rng_key, X, i+1, batch_size, maxiter)
            #     list_params.append(param)
            #     id_print(var)
            #     rng_key, ks = jax.random.split(rng_key)
            #     states, info = inference_loop(ks, state, step_fn, batch_size * batch_iter, param)
            #     X = jax.tree_map(lambda x, y: jnp.concatenate([x, y]), X, states.position)
            # param, var = optimize2(param, kld_warm, build_check(n_epoch), optim, rng_key, X, n_epoch, batch_size, maxiter)
            # list_params.append(param)
            # return (state, param), list_params

            def one_epoch(carry, key):
                state, param, X = carry
                param, err = optimize(param, forward, optim, tol, maxiter, key, X, batch_iter, batch_size)
                ks, kc = jax.random.split(key)
                states, info = inference_loop(ks, state, step_fn, batch_size * batch_iter, param)
                X = jax.tree_map(lambda x, y: jax.random.choice(kc, jnp.concatenate([x, y]), (batch_iter * batch_size,), False), states.position, X)
                return (state, param, X), err
            rng_keys = jax.random.split(rng_key, n_epoch)
            
            # #starting from non-reverse parameters
            # p = ravel_pytree(state.position)[0]
            # param_ = param_init(kc, p)
            (state, param, X), err = jax.lax.scan(one_epoch, (state, param, X), rng_keys[:-1])
            param, err_ = optimize(param, forward, optim, tol, maxiter, rng_keys[-1], X, batch_iter, batch_size)
            return (state, param), jnp.hstack([err, err_])
        
        return SamplingAlgorithm(init_fn, step_fn), warm_fn


class neutra:
    def __new__(
        cls,
        logprob_fn: Callable,
        optim: Optim, d, param_init, flow, reverse,
    ) -> SamplingAlgorithm:

        def init_fn(
            rng: PRNGKey, position: PyTree, 
            batch_size: int, batch_iter: int, tol: float, maxiter : int,
        ):
            p, unraveler_fn = ravel_pytree(position)
            ku, kp = jax.random.split(rng)
            U = jax.vmap(unraveler_fn)(jax.random.normal(ku, shape=(batch_iter * batch_size, d)))

            init_param = param_init(kp, p)
            param, err = optimize(init_param, reverse, optim, tol, maxiter, kp, U, batch_iter, batch_size)
            def pullback_fn(u):
                x, _, ldj = flow(u, jnp.zeros(d), param)
                return logprob_fn(x) + ldj
            push_fn = jax.vmap(lambda u: flow(u, jnp.zeros(d), param)[0])

            return pullback_fn, push_fn, param, err

        return init_fn


def ellipsis(p, m, theta, mu=0.):
    x, unraveler = ravel_pytree(p)
    return (
        unraveler((x - mu) * jnp.cos(theta) + (m - mu) * jnp.sin(theta) + mu),
        (m - mu) * jnp.cos(theta) - (x - mu) * jnp.sin(theta) + mu
    )

def tess_proposal(
    slice_fn: Callable, 
    momentum_generator: Callable,
    T: Callable, T_inv: Callable,
) -> Callable:

    def generate(rng_key: PRNGKey, state: SliceState) -> Tuple[SliceState, SliceInfo]:
        position, _ = state
        kmomentum, kunif, ktheta = jax.random.split(rng_key, 3)
        # step 1: sample momentum
        momentum = momentum_generator(kmomentum, position)
        #step 2-3: get slice (y)
        # logy = slice_fn(position, momentum) + jnp.log(jax.random.uniform(kunif))
        #step 4: get u
        u_position, momentum = T_inv(position, momentum)
        logy = slice_fn(u_position, momentum) + jnp.log(jax.random.uniform(kunif))
        #step 5-6: get theta (ellipsis move), set inital interval
        theta = 2 * jnp.pi * jax.random.uniform(ktheta)
        theta_min = theta - 2 * jnp.pi
        theta_max = theta
        #step 7-8: proposal
        u, m = ellipsis(u_position, momentum, theta)
        #step 9: get new position
        slice = slice_fn(u, m)
        p, m = T(u, m)
        #step 10-20: acceptance
        # slice = slice_fn(p, m)

        def while_fun(vals):
            rng, _, subiter, theta, theta_min, theta_max, *_ = vals
            rng, thetak = jax.random.split(rng)
            theta = jax.random.uniform(thetak, minval=theta_min, maxval=theta_max)
            u, m = ellipsis(u_position, momentum, theta)
            slice = slice_fn(u, m)
            p, m = T(u, m)
            # slice = slice_fn(p, m)
            theta_min = jnp.where(theta < 0, theta, theta_min)
            theta_max = jnp.where(theta > 0, theta, theta_max)
            subiter += 1
            return rng, slice, subiter, theta, theta_min, theta_max, p, m

        _, slice, subiter, theta, *_, position, momentum = jax.lax.while_loop(
            lambda vals: (vals[1] <= logy) | jnp.isinf(ravel_pytree(vals[-2])[0]).any(), 
            while_fun, 
            (rng_key, slice, 1, theta, theta_min, theta_max, p, m)
        )
        return (SliceState(position, slice), 
            SliceInfo(momentum, theta, subiter))

    return generate


def ellipsis2(p, m, theta, mu=0.):
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
        p, m = ellipsis2(position, momentum, theta)
        #step 5: acceptance
        loglikelihood = loglikelihood_fn(p)

        def while_fun(vals):
            rng, _, subiter, theta, theta_min, theta_max, *_ = vals
            rng, thetak = jax.random.split(rng)
            theta = jax.random.uniform(thetak, minval=theta_min, maxval=theta_max)
            p, m = ellipsis2(position, momentum, theta)
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
