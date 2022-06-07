import warnings

import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.example_libraries import stax

from jax.experimental.host_callback import id_print

import pandas as pd

from numpyro.diagnostics import print_summary

from flows import inverse_autoreg, coupling_dense
from distances import kullback_liebler, renyi_alpha
from kernels import atransp_elliptical_slice, elliptical_slice, neutra
from mcmc_utils import inference_loop, inference_loop0

import blackjax


non_lins = {
    'tanh': jax.nn.tanh,
    'elu': jax.nn.elu,
    'relu': jax.nn.relu,
    'swish': jax.nn.swish,
}

flows = {
    'iaf': lambda n, f, h, nl: inverse_autoreg(n, f, h, nl, False),
    'riaf': lambda n, f, h, nl: inverse_autoreg(n, f, h, nl, True),
    'cdense': lambda n, f, h, nl: coupling_dense(n, f, h, nl, False),
    'ncdense': lambda n, f, h, nl: coupling_dense(n, f, h, nl, True),
}

distances = {
    'kld': kullback_liebler,
    'ralpha=0.5': lambda fn, n, f, fi: renyi_alpha(fn, n, f, fi, .5),
    'ralpha=2': lambda fn, n, f, fi: renyi_alpha(fn, n, f, fi, 2.),
    'ralpha=0': lambda fn, n, f, fi: renyi_alpha(fn, n, f, fi, 0.),
}

### ATESS no transformation

def run_ess(
    rng_key, logprob_fn, init_params,
    n, n_warm, n_iter, n_chain,
):
    print(f"\n(T)ESS w/ {n_chain} chains - {n_warm} warmup - {n_iter} samples...")

    tic1 = pd.Timestamp.now()
    ess = elliptical_slice(logprob_fn, n)
    def one_chain(ksam, init_x):
        kwarm, ksam = jrnd.split(ksam)
        state = ess.init(init_x)
        states, info = inference_loop0(kwarm, state, ess.step, n_warm)
        state = jax.tree_map(lambda x: x[-1], states)
        states, info = inference_loop0(ksam, state, ess.step, n_iter)
        return states.position, info.subiter.mean()
    ksam = jrnd.split(rng_key, n_chain)
    samples, subiter = jax.vmap(one_chain)(ksam, init_params)
    # samples, subiter = jax.pmap(one_chain)(ksam, init_params)
    tic2 = pd.Timestamp.now()

    print_summary(samples)
    print("Runtime for (T)ESS", tic2 - tic1)
    return samples

### ATESS

def run_atess(
    rng_key, logprob_fn, init_params, flow, distance,
    optim, n, n_flow, n_hidden, non_linearity,
    n_iter, n_chain,
    n_epochs, batch_size, batch_iter, tol, maxiter,
):
    if flow in ['iaf', 'riaf'] and any([nh != n for nh in n_hidden]):
        warnings.warn('IAF flows always have dimension of hidden units same as params.')
    
    print(f"\nATESS w/ {n_flow} flows (flow: {flow}) - hidden layers={n_hidden} - {non_linearity} nonlinearity")
    print(f"warmup & precond: {n_epochs} epochs - batches of size {batch_size} over {batch_iter} iter - tolerance {tol}")
    print(f"sampling: {n_chain} chains - {n_iter} samples each...")

    tic1 = pd.Timestamp.now()
    param_init, flow, flow_inv = flows[flow](n, n_flow, n_hidden, non_lins[non_linearity])
    reverse, forward = distances[distance](logprob_fn, n, flow, flow_inv)
    atess, warm_fn = atransp_elliptical_slice(logprob_fn, optim, n, param_init, flow, flow_inv, reverse, forward)
    def one_chain(ksam, init_x):
        kinit, kwarm, ksam = jrnd.split(ksam, 3)
        state, param, err = atess.init(kinit, init_x, batch_size, batch_iter, tol, maxiter)
        id_print(err)
        if n_epochs > 0:
            (state, param), error = warm_fn(kwarm, state, param, n_epochs, batch_size, batch_iter, tol, maxiter)
            id_print(error)
        states, info = inference_loop(ksam, state, atess.step, n_iter, param)
        return states.position, info.subiter.mean(), param
    ksam = jrnd.split(rng_key, n_chain)
    samples, subiter, params = jax.vmap(one_chain)(ksam, init_params)
    print(subiter)
    # samples, subiter, diagnose = jax.pmap(one_chain)(ksam, init_params)
    tic2 = pd.Timestamp.now()

    flow_samples = get_flow_samples(ksam[-1], flow, params, n_chain, n_iter, n, init_params)

    print_summary(samples)
    print_summary(flow_samples)
    print("Runtime for ATESS", tic2 - tic1)
    return samples, flow_samples

### NeuTra

def run_neutra(
    rng_key, logprob_fn, init_params, flow, distance,
    optim, n, n_flow, n_hidden, non_linearity,
    n_warm, n_iter, n_chain, nuts,
    batch_size, batch_iter, tol, maxiter, 
):
    if flow in ['iaf', 'riaf'] and any([nh != n for nh in n_hidden]):
        warnings.warn('IAF flows always have dimension of hidden units same as params.')
    
    if flow in ['ncdense', 'cdense']:
        warnings.warn('NeuTra samples for fully coupled dense flows including latent parameters are irrelevant, see code.')

    print(f"\nNeuTra w/ {n_flow} (flow: {flow}) flows - hidden layers={n_hidden} - {non_linearity} nonlinearity")
    print(f"precond: batches of {batch_size} atoms over {batch_iter} iter - tolerance {tol}")
    print(f"sampling: {n_chain} chains - {n_warm} warmup - {n_iter} samples...")

    tic1 = pd.Timestamp.now()
    param_init, flow, flow_inv = flows[flow](n, n_flow, n_hidden, non_lins[non_linearity])
    reverse, _ = distances[distance](logprob_fn, n, flow, flow_inv)
    init_fn = neutra(logprob_fn, optim, n, param_init, flow, reverse)
    def one_chain(ksam, init_x):
        kinit, kwarm, ksam = jrnd.split(ksam, 3)
        pullback_fn, push_fn, param, err = init_fn(kinit, init_x, batch_size, batch_iter, tol, maxiter)
        id_print(err)
        state, kernel = run_hmc_warmup(kwarm, pullback_fn, init_x, n_warm, .8, True, nuts=nuts)
        states, info = inference_loop0(ksam, state, kernel, n_iter)
        return push_fn(states.position), param
    ksam = jrnd.split(rng_key, n_chain)
    samples, params = jax.vmap(one_chain)(ksam, init_params)
    # samples, info = jax.pmap(one_chain)(ksam, init_params)
    tic2 = pd.Timestamp.now()

    flow_samples = get_flow_samples(ksam[-1], flow, params, n_chain, n_iter, n, init_params)

    print_summary(samples)
    print_summary(flow_samples)
    print("Runtime for NeuTra", tic2 - tic1)
    return samples, flow_samples

def get_flow_samples(rng_key, flow, params, n_chain, n_iter, d, init_params):
    def one_chain(ksam, chain_info):
        init_x, param = chain_info
        ku, kv = jax.random.split(ksam)
        _, unraveler_fn = jax.flatten_util.ravel_pytree(init_x)
        U = jax.vmap(unraveler_fn)(jax.random.normal(ku, shape=(n_iter, d)))
        V = jax.random.normal(kv, shape=(n_iter, d))
        X, *_ = jax.vmap(flow, (0, 0, None))(U, V, param)
        return X
    ksam = jrnd.split(rng_key, n_chain)
    samples = jax.vmap(one_chain)(ksam, (init_params, params))
    # samples = jax.pmap(one_chain)(ksam, (init_params, params))
    return samples

### NUTS

def run_nuts(
    rng_key, logprob_fn, init_params,
    n_warm, n_iter, n_chain,
):
    print(f"\nNUTS w/ {n_chain} chains - {n_warm} warmup - {n_iter} samples...")

    tic1 = pd.Timestamp.now()
    def one_chain(ksam, init_param):
        kwarm, ksam = jrnd.split(ksam)
        # id_print(jnp.zeros(1))
        state, kernel = run_hmc_warmup(kwarm, logprob_fn, init_param, n_warm, .8, True, nuts=True)
        # id_print(jnp.zeros(1)+1)
        states, info = inference_loop0(ksam, state, kernel, n_iter)
        # id_print(jnp.zeros(1)+2)
        return states.position, info
    ksam = jrnd.split(rng_key, n_chain)
    samples, info = jax.vmap(one_chain)(ksam, init_params)
    # samples, info = jax.pmap(one_chain)(ksam, init_params)
    tic2 = pd.Timestamp.now()

    print_summary(samples)
    print("Runtime for NUTS", tic2 - tic1)
    return samples

### HMC

def run_hmc(
    rng_key, logporb_fn, init_params,
    n_warm, n_iter, n_chain,
):
    def one_chain(ksam, init_param):
        kwarm, ksam = jrnd.split(ksam)
        state, kernel = run_hmc_warmup(kwarm, logporb_fn, init_param, n_warm, .65, True, nuts=False)
        states, info = inference_loop0(ksam, state, kernel, n_iter)
        return states.position, info
    ksam = jrnd.split(rng_key, n_chain)
    samples, info = jax.vmap(one_chain)(ksam, init_params)
    print_summary(samples)
    return samples

### HMC Tuning

def run_hmc_warmup(
    rng_key, logprob_fn, init_param,
    n_warm, target_acceptance_rate, is_mass_matrix_diagonal,
    nuts=False, progress_bar=False
):
    warmup = blackjax.window_adaptation(
        algorithm=blackjax.nuts if nuts else blackjax.hmc,
        logprob_fn=logprob_fn,
        num_steps=n_warm,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        target_acceptance_rate=target_acceptance_rate,
        progress_bar=progress_bar,
    )
    state, kernel, _ = warmup.run(rng_key, init_param)
    return state, kernel
