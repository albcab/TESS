import argparse

import pandas as pd

import jax.numpy as jnp
import jax.random as jrnd
from jax.example_libraries import stax
import optax

from distributions import Banana
from exec import run_atess, run_neutra, run_nuts

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


N_PARAM = 2

def main(args):

    print("Setting up Banana density...")
    dist = Banana()

    [n_chain, n_warm, n_iter] = args.sampling_param
    ksam, kinit = jrnd.split(jrnd.PRNGKey(args.seed))
    dist.initialize_model(kinit, n_chain)

    print(f"\nNUTS w/ {n_chain} chains - {n_warm} warmup - {n_iter} samples...")
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
    maxiter = args.max_iter
    print(f"\nUsing constant schedule {schedule} & maxiter {maxiter}.")

    n_epochs = args.epoch
    [batch_iter, batch_size] = args.batch_shape
    tol = args.tol

    optim = optax.adam(schedule)
    non_lin = 'tanh'
    n_hidden = [N_PARAM] * 1
    n_flow = 2
    norm = False
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)

    optim = optax.adam(schedule)
    non_lin = 'elu'
    n_hidden = [N_PARAM] * 1
    n_flow = 2
    norm = False
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)

    optim = optax.adam(schedule)
    non_lin = 'elu'
    n_hidden = [N_PARAM] * 1
    n_flow = 2
    norm = True
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)
    
    optim = optax.adam(schedule)
    non_lin = 'tanh'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    norm = False
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)

    optim = optax.adam(schedule)
    non_lin = 'elu'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    norm = True
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)

    optim = optax.adam(schedule)
    non_lin = 'tanh'
    n_hidden = [20] * 2
    n_flow = 1
    norm = False
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)

    optim = optax.adam(schedule)
    non_lin = 'tanh'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    run_neutra(ksam, dist.logprob_fn, dist.init_params, 
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_warm, n_iter, n_chain, True, batch_size, batch_iter, tol, maxiter)

    optim = optax.adam(schedule)
    non_lin = 'elu'
    n_hidden = [N_PARAM] * 1
    n_flow = 2
    run_neutra(ksam, dist.logprob_fn, dist.init_params, 
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_warm, n_iter, n_chain, True, batch_size, batch_iter, tol, maxiter)

    optim = optax.adam(schedule)
    non_lin = 'elu'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    run_neutra(ksam, dist.logprob_fn, dist.init_params, 
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_warm, n_iter, n_chain, True, batch_size, batch_iter, tol, maxiter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('-t', '--tol', type=float, default=1e-0)
    parser.add_argument('-m', '--max-iter', type=int, default=1e6)
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-b', '--batch-shape', type=int, nargs=2, default=[10, 10_000])
    parser.add_argument(
        "-s", "--sampling-param", type=int, nargs=2,
        help="Sampling parameters [n_chain, n_warm, n_iter]",
        default=[4, 1000, 1000]
    )
    args = parser.parse_args()
    main(args)