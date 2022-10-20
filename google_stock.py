import argparse

import pandas as pd

import jax
import optax

from distributions import RegimeSwitchHMM
from execute import run

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


N_PARAM = 9

def main(args):

    print("Loading Google stock data...")
    data = pd.read_csv('google.csv')
    y = data.dl_ac.values * 100
    T, _ = data.shape
    
    print("Setting up Regime switching hidden Markov model...")
    dist = RegimeSwitchHMM(T, y)

    [n_warm, n_iter] = args.sampling_param
    schedule = optax.exponential_decay(init_value=1e-2,
        transition_steps=n_warm-10, decay_rate=.1, transition_begin=10)
    optim = optax.adam(schedule)

    run(dist, args, optim, N_PARAM, batch_fn=jax.vmap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('-m', '--max-iter', type=int, default=1)
    parser.add_argument('-b', '--batch-shape', type=int, nargs=2, default=[4, 32])
    parser.add_argument('-nl', '--non-linearity', type=str, default='relu')
    parser.add_argument('-f', '--flow', type=str, default='coupling')
    parser.add_argument('-d', '--distance', type=str, default='kld')
    parser.add_argument('-nh', '--n-hidden', type=int, default=2)
    parser.add_argument('-nf', '--n-flow', type=int, default=2)
    parser.add_argument('-nb', '--num-bins', type=int, default=None)
    parser.add_argument(
        "-s", "--sampling-param", type=int, nargs=3,
        help="Sampling parameters [n_warm, n_iter]",
        default=[400, 100]
    )
    parser.add_argument('-np', '--preconditon_iter', type=int, default=100)
    parser.add_argument('-s1', '--init_step_size', type=float, default=None)
    parser.add_argument('-s2', '--p_init_step_size', type=float, default=None)
    args = parser.parse_args()
    main(args)