import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.example_libraries import stax

from jax.experimental.host_callback import id_print

import pandas as pd

from numpyro.diagnostics import print_summary

from kernels import atransp_elliptical_slice, elliptical_slice, neutra
from mcmc_utils import inference_loop, inference_loop0

import blackjax


non_lins = {
    'tanh': jax.nn.tanh,
    'elu': jax.nn.elu,
    'relu': jax.nn.relu,
    'swish': jax.nn.swish,
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
    rng_key, logprob_fn, init_params,
    optim, n, n_flow, n_hidden, non_linearity, norm,
    n_iter, n_chain,
    n_epochs, batch_size, batch_iter, tol, maxiter,
):
    print(f"\nATESS w/ {n_flow} flows - hidden layers={n_hidden} - {non_linearity} nonlinearity - batch normalization? {norm},")
    print(f"warmup & precond: {n_epochs} epochs - {batch_size} samples over {batch_iter} iter ({int(batch_size/batch_iter)} samples per iter) - tolerance {tol},")
    print(f"sampling: {n_chain} chains - {n_iter} samples each...")

    tic1 = pd.Timestamp.now()
    atess, warm_fn = atransp_elliptical_slice(logprob_fn, optim, n, n_flow, n_hidden, non_lins[non_linearity], norm)
    def one_chain(ksam, init_x):
        kinit, kwarm, ksam = jrnd.split(ksam, 3)
        state, param, err = atess.init(kinit, init_x, batch_size, batch_iter, tol, maxiter)
        id_print(err)
        (state, param), error = warm_fn(kwarm, state, param, n_epochs, batch_size, batch_iter, tol, maxiter)
        id_print(error)
        states, info = inference_loop(ksam, state, atess.step, n_iter, param)
        return states.position, info.subiter.mean(), (err, param, error)
    ksam = jrnd.split(rng_key, n_chain)
    samples, subiter, diagnose = jax.vmap(one_chain)(ksam, init_params)
    # samples, subiter, diagnose = jax.pmap(one_chain)(ksam, init_params)
    tic2 = pd.Timestamp.now()

    print_summary(samples)
    print("Runtime for ATESS", tic2 - tic1)
    return samples

### NeuTra

def run_neutra(
    rng_key, logprob_fn, init_params,
    optim, n, n_flow, n_hidden, non_linearity,
    n_warm, n_iter, n_chain, nuts,
    batch_size, batch_iter, tol, maxiter, invert
):
    print(f"\nNeuTra w/ {n_flow} (reverse: {invert}) flows - hidden layers={n_hidden} - {non_linearity} nonlinearity,")
    print(f"precond: {batch_size} atoms over {batch_iter} iter ({int(batch_size/batch_iter)} atoms per iter) - tolerance {tol},")
    print(f"sampling: {n_chain} chains - {n_warm} warmup - {n_iter} samples...")

    tic1 = pd.Timestamp.now()
    init_fn = neutra(logprob_fn, optim, n, n_flow, n_hidden, non_lins[non_linearity], invert)
    def one_chain(ksam, init_x):
        kinit, kwarm, ksam = jrnd.split(ksam, 3)
        pullback_fn, push_fn, err = init_fn(kinit, init_x, batch_size, batch_iter, tol, maxiter)
        id_print(err)
        state, kernel = run_hmc_warmup(kwarm, pullback_fn, init_x, n_warm, .8, True, nuts=nuts)
        states, info = inference_loop0(ksam, state, kernel, n_iter)
        return push_fn(states.position), info
    ksam = jrnd.split(rng_key, n_chain)
    samples, info = jax.vmap(one_chain)(ksam, init_params)
    # samples, info = jax.pmap(one_chain)(ksam, init_params)
    tic2 = pd.Timestamp.now()

    print_summary(samples)
    print("Runtime for NeuTra", tic2 - tic1)
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
