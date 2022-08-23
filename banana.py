import argparse
import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=10'

import jax
# print(jax.devices())
import optax

from distributions import Banana
from execute import full_run, run

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


N_PARAM = 2

def main(args):

    print("Setting up Banana density...")
    dist = Banana()

    [n_warm, n_iter] = args.sampling_param
    schedule = optax.exponential_decay(init_value=1e-2,
        transition_steps=n_warm-10, decay_rate=.1, transition_begin=10)
    optim = optax.adam(schedule)

    full_run(dist, args, optim, N_PARAM, batch_fn=jax.vmap)

    # jnp.savez(f'banana_{flow}_{distance}_{non_lin}_{n_epochs}.npz', 
    #     ess_samples=ess_samples, nuts_samples=nuts_samples,
    #     atess_samples=atess_samples, atess_flow_samples=atess_flow_samples,
    #     neutra_samples=neutra_samples, neutra_flow_samples=neutra_flow_samples,
    # )

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
    args = parser.parse_args()
    main(args)