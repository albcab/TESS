import argparse
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=10'

import warnings

import jax.numpy as jnp
import jax.random as jrnd
import optax

from distributions import NealsFunnel
from exec import run_atess, run_neutra, run_nuts, run_ess

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

import jax
print(jax.devices())


N_PARAM = 2

def main(args):

    if args.flow in ['iaf', 'riaf'] and args.n_hidden_units != N_PARAM:
        warnings.warn('IAF flows always have dimension of hidden units same as params.')

    if args.flow in ['ncdense', 'cdense']:
        warnings.warn('NeuTra samples for fully coupled dense flows including latent parameters are irrelevant, see code.')

    print("Setting up Neal's Funnel density...")
    dist = NealsFunnel()

    [n_chain, n_warm, n_iter] = args.sampling_param
    ksam, kinit = jrnd.split(jrnd.PRNGKey(args.seed))
    dist.initialize_model(kinit, n_chain)

    ess_samples = run_ess(ksam, dist.logprob_fn, dist.init_params, 
        N_PARAM, n_warm, n_iter, n_chain)

    nuts_samples = run_nuts(ksam, dist.logprob_fn, dist.init_params, 
        n_warm, n_iter, n_chain)

    # schedule = optax.polynomial_schedule(1e-2, 1e-4, 1, 9000, 1000)
    # schedule = optax.piecewise_constant_schedule(init_value=1e-2,
    #     boundaries_and_scales={2000: .1, 8000: .1})
    schedule = optax.exponential_decay(init_value=1e-2,
        transition_steps=4e3, decay_rate=.1, transition_begin=1e3)
    # schedule = 1e-3
    maxiter = args.max_iter
    # print(f"\nUsing constant schedule {schedule} & maxiter {maxiter}.")
    print(f"\nUsing exponential decay schedule & maxiter {maxiter}.")
    optim = optax.adam(schedule)

    n_epochs = args.epoch
    [batch_iter, batch_size] = args.batch_shape
    tol = args.tol
    flow = args.flow
    non_lin = args.non_linearity
    n_hidden = [args.n_hidden_units] * args.n_hidden
    n_flow = args.n_flow

    atess_samples, atess_flow_samples = run_atess(ksam, dist.logprob_fn, dist.init_params, flow,
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)

    neutra_samples, neutra_flow_samples = run_neutra(ksam, dist.logprob_fn, dist.init_params, flow,
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_warm, n_iter, n_chain, True, batch_size, batch_iter, tol, maxiter)

    jnp.savez(f'nealsfunnel_{flow}_{non_lin}_{n_epochs}.npz', 
        ess_samples=ess_samples, nuts_samples=nuts_samples,
        atess_samples=atess_samples, atess_flow_samples=atess_flow_samples,
        neutra_samples=neutra_samples, neutra_flow_samples=neutra_flow_samples,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('-t', '--tol', type=float, default=1e-1)
    parser.add_argument('-m', '--max-iter', type=int, default=1.1e4)
    parser.add_argument('-e', '--epoch', type=int, default=5)
    parser.add_argument('-b', '--batch-shape', type=int, nargs=2, default=[20, 128])
    parser.add_argument('-nl', '--non-linearity', type=str, default='tanh')
    parser.add_argument('-f', '--flow', type=str, default='riaf')
    parser.add_argument('-nh', '--n-hidden', type=int, default=2)
    parser.add_argument('-nf', '--n-flow', type=int, default=2)
    parser.add_argument('-nhu', '--n-hidden-units', type=int, default=N_PARAM)
    parser.add_argument(
        "-s", "--sampling-param", type=int, nargs=3,
        help="Sampling parameters [n_chain, n_warm, n_iter]",
        default=[10, 2000, 10_000]
    )
    args = parser.parse_args()
    main(args)