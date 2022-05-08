import argparse

import pandas as pd

import jax.numpy as jnp
import jax.random as jrnd
from jax.example_libraries import stax
import optax

from distributions import Banana
from exec import run_atess, run_neutra, run_nuts

from jax.config import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


N_PARAM = 2
SEED = 0

def main():

    print("Setting up Banana density...")
    dist = Banana()

    n_warm = 1000
    n_iter = 1000
    n_chain = 4
    print(f"Sampling {n_chain} chains for {n_iter} iterations...")
    ksam, kinit = jrnd.split(jrnd.PRNGKey(SEED))
    dist.initialize_model(kinit, n_chain)

    print(f"\nRunning NUTS w/ {n_warm} warmup (tuning) iter...")
    tic1 = pd.Timestamp.now()
    run_nuts(ksam, dist.logprob_fn, dist.init_params, 
        n_warm, n_iter, n_chain)
    tic2 = pd.Timestamp.now()
    print("Runtime for NUTS", tic2 - tic1)

    # schedule = optax.polynomial_schedule(1e-2, 1e-4, 1, 9000, 1000)
    # schedule = optax.piecewise_constant_schedule(init_value=1e-2,
    #     boundaries_and_scales={1000: .1, 4000: .1})
    schedule = 1e-3
    optim = optax.adam(schedule)
    n_epochs = 5
    batch_size = 10000
    n_atoms = 4096
    vi_iter = 10
    print(f"\nUsing linear learning rate schedule w/ {n_atoms} pre-atoms, {vi_iter} iter, {n_epochs} warmup epochs of {batch_size} size each...")

    optim = optax.adam(schedule)
    non_lin = stax.Tanh
    n_hidden = [N_PARAM] * 1
    n_flow = 2
    print(f"Running ATESS w/ {n_flow} concatenated flows each w/ hidden layers={n_hidden} and Tanh nonlinearity...")
    tic1 = pd.Timestamp.now()
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_atoms, vi_iter, n_epochs, batch_size,
        n_iter, n_chain,)
    tic2 = pd.Timestamp.now()
    print("Runtime for ATESS", tic2 - tic1)

    optim = optax.adam(schedule)
    non_lin = stax.Tanh
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    print(f"Running ATESS w/ {n_flow} concatenated flows each w/ hidden layers={n_hidden} and Tanh nonlinearity...")
    tic1 = pd.Timestamp.now()
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_atoms, vi_iter, n_epochs, batch_size,
        n_iter, n_chain,)
    tic2 = pd.Timestamp.now()
    print("Runtime for ATESS", tic2 - tic1)

    optim = optax.adam(schedule)
    non_lin = stax.Tanh
    n_hidden = [20] * 2
    n_flow = 1
    print(f"Running ATESS w/ {n_flow} concatenated flows each w/ hidden layers={n_hidden} and Tanh nonlinearity...")
    tic1 = pd.Timestamp.now()
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_atoms, vi_iter, n_epochs, batch_size,
        n_iter, n_chain,)
    tic2 = pd.Timestamp.now()
    print("Runtime for ATESS", tic2 - tic1)

    schedule = optax.polynomial_schedule(1e-2, 1e-4, 1, 9000, 1000)
    # schedule = optax.piecewise_constant_schedule(init_value=1e-2,
    #     boundaries_and_scales={1000: .1, 4000: .1})
    optim = optax.adam(schedule)
    n_atoms = 4096
    vi_iter = 5000
    print(f"\nUsing schedule as NeuTra w/ {n_warm} warmup (tuning) iter w/ {n_atoms} atoms and {vi_iter} preconditioning iter...")

    optim = optax.adam(schedule)
    non_lin = stax.Tanh
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    print(f"Running NeuTra w/ {n_flow} concatenated flows each w/ hidden layers={n_hidden} and Tanh nonlinearity...")
    tic1 = pd.Timestamp.now()
    run_neutra(ksam, dist.logprob_fn, dist.init_params, 
        optim, N_PARAM, n_flow, n_hidden, non_lin, 
        n_atoms, vi_iter, n_warm, n_iter, n_chain, nuts=True)
    tic2 = pd.Timestamp.now()
    print("Runtime for NeuTra", tic2 - tic1)

    optim = optax.adam(schedule)
    non_lin = stax.Elu
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    print(f"Running NeuTra w/ {n_flow} concatenated flows each w/ hidden layers={n_hidden} and Elu nonlinearity...")
    tic1 = pd.Timestamp.now()
    run_neutra(ksam, dist.logprob_fn, dist.init_params, 
        optim, N_PARAM, n_flow, n_hidden, non_lin, 
        n_atoms, vi_iter, n_warm, n_iter, n_chain, nuts=True)
    tic2 = pd.Timestamp.now()
    print("Runtime for NeuTra", tic2 - tic1)


if __name__ == "__main__":
    main()