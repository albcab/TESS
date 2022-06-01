import argparse
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=10'

import jax.numpy as jnp
import jax.random as jrnd
import optax

from distributions import Banana
from exec import run_atess, run_neutra, run_nuts, run_ess

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

import jax
print(jax.devices())


N_PARAM = 2

def main(args):

    print("Setting up Banana density...")
    dist = Banana()

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
    schedule = optax.exponential_decay(init_value=1e-3,
        transition_steps=4e3, decay_rate=.1, transition_begin=1e3)
    # schedule = 1e-3
    optim = optax.adam(schedule)
    maxiter = args.max_iter
    # print(f"\nUsing constant schedule {schedule} & maxiter {maxiter}.")
    print(f"\nUsing exponential decay schedule & maxiter {maxiter}.")

    n_epochs = args.epoch
    [batch_iter, batch_size] = args.batch_shape
    tol = args.tol

    # non_lin = 'tanh'
    # n_hidden = [N_PARAM] * 1
    # n_flow = 2
    # norm = False
    # atess_2x1_samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
    #     optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
    #     n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)
    
    non_lin = 'tanh'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    norm = False
    atess_4x2_samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
        n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)

    # non_lin = 'tanh'
    # n_hidden = [20] * 2
    # n_flow = 1
    # norm = False
    # atess_20_samples = run_atess(ksam, dist.logprob_fn, dist.init_params,
    #     optim, N_PARAM, n_flow, n_hidden, non_lin, norm,
    #     n_iter, n_chain, n_epochs, batch_size, batch_iter, tol, maxiter)

    non_lin = 'tanh'
    n_hidden = [N_PARAM] * 2
    n_flow = 4
    invert = True
    neutra_tanh_samples = run_neutra(ksam, dist.logprob_fn, dist.init_params, 
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_warm, n_iter, n_chain, True, batch_size, batch_iter, tol, maxiter, invert)

    # non_lin = 'elu'
    # n_hidden = [N_PARAM] * 2
    # n_flow = 4
    # neutra_elu_samples = run_neutra(ksam, dist.logprob_fn, dist.init_params, 
    #     optim, N_PARAM, n_flow, n_hidden, non_lin,
    #     n_warm, n_iter, n_chain, True, batch_size, batch_iter, tol, maxiter)

    jnp.savez('banana.npz', 
        ess_samples=ess_samples, nuts_samples=nuts_samples,
        atess_4x2_samples=atess_4x2_samples, #atess_2x1_samples=atess_2x1_samples, atess_20_samples=atess_20_samples,
        neutra_tanh_samples=neutra_tanh_samples, #neutra_elu_samples=neutra_elu_samples,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('-t', '--tol', type=float, default=1e-1)
    parser.add_argument('-m', '--max-iter', type=int, default=1.1e4)
    parser.add_argument('-e', '--epoch', type=int, default=5)
    # parser.add_argument('-b', '--batch-shape', type=int, nargs=2, default=[100, 10_000])
    parser.add_argument('-b', '--batch-shape', type=int, nargs=2, default=[1, 5_000])
    parser.add_argument(
        "-s", "--sampling-param", type=int, nargs=3,
        help="Sampling parameters [n_chain, n_warm, n_iter]",
        default=[10, 2000, 10_000]
    )
    args = parser.parse_args()
    main(args)