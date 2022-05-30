import argparse
from typing import Tuple

import pandas as pd
import numpy as np

import jax.numpy as jnp
import jax.random as jrnd
from jax.example_libraries import stax
import optax

from distributions import HorseshoeLogisticReg
from exec import run_atess, run_neutra, run_nuts, run_ess

from numpyro.infer import Predictive
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


def main(args):

    print("Loading German credit data...")
    data = pd.read_table('german.data-numeric', header=None, delim_whitespace=True)
    ### Pre processing data as in NeuTra paper
    y = -1 * (data.iloc[:, -1].values - 2)
    X = data.iloc[:, :-1].apply(lambda x: -1 + (x - x.min()) * 2 / (x.max() - x.min()), axis=0).values
    # X = data.iloc[:, :-1].apply(lambda x: (x - x.mean()) / x.std(), axis=0).values
    X = np.concatenate([np.ones((1000, 1)), X], axis=1)
    N_OBS, N_REG = X.shape
    N_PARAM = N_REG * 2 + 1
    
    print("Setting up German credit logistic horseshoe model...")
    dist = HorseshoeLogisticReg(X, y)

    [n_chain, n_warm, n_iter] = args.sampling_param
    ksam, kinit = jrnd.split(jrnd.PRNGKey(args.seed))
    dist.initialize_model(kinit, n_chain)

    run_ess(ksam, dist.logprob_fn, dist.init_params, 
        N_PARAM, n_warm, n_iter, n_chain)

    run_nuts(ksam, dist.logprob_fn, dist.init_params, 
        n_warm, n_iter, n_chain)

    # schedule = optax.polynomial_schedule(1e-2, 1e-4, 1, 9000, 1000)
    # schedule = optax.piecewise_constant_schedule(init_value=1e-2,
    #     boundaries_and_scales={2000: .1, 8000: .1})
    schedule = optax.exponential_decay(init_value=1e-2,
        transition_steps=7e3, decay_rate=.1, transition_begin=1e4)
    # schedule = 1e-3
    optim = optax.adam(schedule)
    maxiter = args.max_iter
    # print(f"\nUsing constant schedule {schedule} & maxiter {maxiter}.")
    print(f"\nUsing exponential decay schedule & maxiter {maxiter}.")

    n_epochs = args.epoch
    [batch_iter, batch_size] = args.batch_shape
    tol = args.tol

    non_lin = 'tanh'
    n_hidden = [N_PARAM] * 1
    n_flow = 6
    norm = True
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)

    non_lin = 'elu'
    n_hidden = [N_PARAM] * 1
    n_flow = 6
    norm = True
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)
    
    non_lin = 'tanh'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    norm = True
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)

    non_lin = 'elu'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    norm = True
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)

    non_lin = 'tanh'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    norm = False
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)

    non_lin = 'elu'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    norm = False
    samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)

    non_lin = 'tanh'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    run_neutra(ksam, dist.logprob_fn, dist.init_params, 
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_warm, n_iter, n_chain, True, batch_size, batch_iter, tol, maxiter)

    non_lin = 'elu'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    run_neutra(ksam, dist.logprob_fn, dist.init_params, 
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_warm, n_iter, n_chain, True, batch_size, batch_iter, tol, maxiter)
    
    non_lin = 'tanh'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    run_neutra(ksam, dist.logprob_fn, dist.init_params, 
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_warm, n_iter, n_chain, True, batch_size, batch_iter, tol, maxiter)

    non_lin = 'elu'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    run_neutra(ksam, dist.logprob_fn, dist.init_params, 
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_warm, n_iter, n_chain, True, batch_size, batch_iter, tol, maxiter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('-t', '--tol', type=float, default=1e-1)
    parser.add_argument('-m', '--max-iter', type=int, default=2e4)
    parser.add_argument('-e', '--epoch', type=int, default=5)
    parser.add_argument('-b', '--batch-shape', type=int, nargs=2, default=[100, 10_000])
    parser.add_argument(
        "-s", "--sampling-param", type=int, nargs=2,
        help="Sampling parameters [n_chain, n_warm, n_iter]",
        default=[1, 1000, 1000]
    )
    args = parser.parse_args()
    main(args)