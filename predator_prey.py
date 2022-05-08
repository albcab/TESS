import argparse
from typing import Tuple

import pandas as pd

import jax.numpy as jnp
import jax.random as jrnd
from jax.example_libraries import stax
import optax

from distributions import PredatorPrey
from exec import run_atess, run_neutra, run_nuts, run_ess

from numpyro.infer import Predictive
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


N_PARAM = 8

def main(args):

    print("Loading predator-prey data...")
    data = pd.read_table("lynxhare.txt", sep=" ", names=['year', 'prey', 'pred', ''])
    # data = data.loc[data.year >= 1920]
    # print(data)
    
    print("Setting up predator-prey model...")
    dist = PredatorPrey(
        data.year.values, data.pred.values, data.prey.values,
        *args.prior_means, *args.prior_scale, 
    )

    n_warm = 1000
    # n_iter = 1000
    # n_chain = 4
    [n_iter, n_chain] = args.sampling_param
    print(f"Sampling {n_chain} chains for {n_iter} iterations...")
    ksam, kinit = jrnd.split(jrnd.PRNGKey(args.seed))
    dist.initialize_model(kinit, n_chain)

    print(f"\nRunning (T)ESS w/ {n_warm} warmup (tuning) iter...")
    tic1 = pd.Timestamp.now()
    run_ess(ksam, dist.logprob_fn, dist.init_params, 
        N_PARAM, n_warm, n_iter, n_chain)
    tic2 = pd.Timestamp.now()
    print("Runtime for (T)ESS", tic2 - tic1)

    # print(f"\nRunning NUTS w/ {n_warm} warmup (tuning) iter...")
    # tic1 = pd.Timestamp.now()
    # run_nuts(ksam, dist.logprob_fn, dist.init_params, 
    #     n_warm, n_iter, n_chain)
    # tic2 = pd.Timestamp.now()
    # print("Runtime for NUTS", tic2 - tic1)

    # # schedule = optax.polynomial_schedule(1e-2, 1e-4, 1, 9000, 1000)
    # schedule = optax.piecewise_constant_schedule(init_value=1e-2,
    #     boundaries_and_scales={1000: .1, 4000: .1})
    # optim = optax.adam(schedule)
    # n_atoms = 4096
    # vi_iter = 5000
    # print(f"\nUsing schedule as NeuTra w/ {n_warm} warmup (tuning) iter w/ {n_atoms} atoms and {vi_iter} preconditioning iter...")

    # # optim = optax.adam(schedule)
    # # non_lin = stax.Tanh
    # # n_hidden = [N_PARAM] * 4
    # # n_flow = 4
    # # print(f"Running NeuTra w/ {n_flow} concatenated flows each w/ hidden layers={n_hidden} and Tanh nonlinearity...")
    # # tic1 = pd.Timestamp.now()
    # # run_neutra(ksam, dist.logprob_fn, dist.init_params, 
    # #     optim, N_PARAM, n_flow, n_hidden, non_lin, 
    # #     n_atoms, vi_iter, n_warm, n_iter, n_chain, nuts=True)
    # # tic2 = pd.Timestamp.now()
    # # print("Runtime for NeuTra", tic2 - tic1)

    # optim = optax.adam(schedule)
    # non_lin = stax.Elu
    # n_hidden = [N_PARAM] * 4
    # n_flow = 4
    # print(f"Running NeuTra w/ {n_flow} concatenated flows each w/ hidden layers={n_hidden} and Elu nonlinearity...")
    # tic1 = pd.Timestamp.now()
    # run_neutra(ksam, dist.logprob_fn, dist.init_params, 
    #     optim, N_PARAM, n_flow, n_hidden, non_lin, 
    #     n_atoms, vi_iter, n_warm, n_iter, n_chain, nuts=True)
    # tic2 = pd.Timestamp.now()
    # print("Runtime for NeuTra", tic2 - tic1)

    # schedule = optax.polynomial_schedule(1e-2, 1e-4, 1, 9000, 1000)
    schedule = optax.piecewise_constant_schedule(init_value=1e-2,
        boundaries_and_scales={2000: .1, 8000: .1})
    optim = optax.adam(schedule)
    n_epochs = args.epoch
    batch_size = args.batch
    n_atoms = args.n_atoms
    vi_iter = 1000
    print(f"\nUsing schedule as NeuTra x 2 w/ {n_atoms} pre-atoms, {vi_iter} iter, {n_epochs} warmup epochs of {batch_size} size each...")

    figures = []

    optim = optax.adam(schedule)
    non_lin = stax.Elu
    n_hidden = [N_PARAM] * 4
    n_flow = 2
    print(f"Running ATESS w/ {n_flow} concatenated flows each w/ hidden layers={n_hidden} and Elu nonlinearity...")
    tic1 = pd.Timestamp.now()
    samples, diagnose = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_atoms, vi_iter, n_epochs, batch_size,
        n_iter, n_chain,)
    tic2 = pd.Timestamp.now()
    print("Runtime for ATESS", tic2 - tic1)

    for i in range(n_chain):
        fig = plt.figure(figsize=(15, 5))
        plt.plot(diagnose[2][i].T)
        plt.legend(jnp.arange(n_epochs))
        figures.append(fig)

    # optim = optax.adam(schedule)
    # non_lin = stax.Elu
    # n_hidden = [N_PARAM] * 2
    # n_flow = 4
    # print(f"Running ATESS w/ {n_flow} concatenated flows each w/ hidden layers={n_hidden} and Elu nonlinearity...")
    # tic1 = pd.Timestamp.now()
    # samples, diagnose = run_atess(ksam, dist.logprob_fn, dist.init_params,
    #     optim, N_PARAM, n_flow, n_hidden, non_lin,
    #     n_atoms, vi_iter, n_epochs, batch_size,
    #     n_iter, n_chain,)
    # tic2 = pd.Timestamp.now()
    # print("Runtime for ATESS", tic2 - tic1)

    # for i in range(n_chain):
    #     fig = plt.figure(figsize=(15, 5))
    #     plt.plot(diagnose[2][i].T)
    #     plt.legend(jnp.arange(n_epochs))
    #     figures.append(fig)

    optim = optax.adam(schedule)
    non_lin = stax.Tanh
    n_hidden = [N_PARAM] * 4
    n_flow = 2
    print(f"Running ATESS w/ {n_flow} concatenated flows each w/ hidden layers={n_hidden} and Tanh nonlinearity...")
    tic1 = pd.Timestamp.now()
    samples, diagnose = run_atess(ksam, dist.logprob_fn, dist.init_params,
        optim, N_PARAM, n_flow, n_hidden, non_lin,
        n_atoms, vi_iter, n_epochs, batch_size,
        n_iter, n_chain,)
    tic2 = pd.Timestamp.now()
    print("Runtime for ATESS", tic2 - tic1)

    for i in range(n_chain):
        fig = plt.figure(figsize=(15, 5))
        plt.plot(diagnose[2][i].T)
        plt.legend(jnp.arange(n_epochs))
        figures.append(fig)

    # optim = optax.adam(schedule)
    # non_lin = stax.Tanh
    # n_hidden = [N_PARAM] * 2
    # n_flow = 2
    # print(f"Running ATESS w/ {n_flow} concatenated flows each w/ hidden layers={n_hidden} and Tanh nonlinearity...")
    # tic1 = pd.Timestamp.now()
    # samples, diagnose = run_atess(ksam, dist.logprob_fn, dist.init_params,
    #     optim, N_PARAM, n_flow, n_hidden, non_lin,
    #     n_atoms, vi_iter, n_epochs, batch_size,
    #     n_iter, n_chain,)
    # tic2 = pd.Timestamp.now()
    # print("Runtime for ATESS", tic2 - tic1)

    # for i in range(n_chain):
    #     fig = plt.figure(figsize=(15, 5))
    #     plt.plot(diagnose[2][i].T)
    #     plt.legend(jnp.arange(n_epochs))
    #     figures.append(fig)

    with PdfPages('google_stock.pdf') as pdf:
        for fig in figures:
            pdf.savefig(fig)






    # [hidden_size, n_layer, n_flow, n_atoms] = args.transport_param
    # n_hidden = [hidden_size] * n_layer
    # optim = args.optimizer(args.learning_rate)
    # # optim = optax.chain(
    # #     optax.clip(.1),
    # #     args.optimizer(args.learning_rate)
    # # )
    # print(f"Using {n_flow} concatenated flows each w/ hidden layers={n_hidden}")

    # [vi_iter, n_epochs, batch_size] = args.warmup_param
    # pre_iter = n_epochs * vi_iter
    # n_warm = n_epochs * batch_size #too much warm up not good for nuts, it semms
    # n_warm = vi_iter
    # [n_iter, n_chain] = args.sampling_param

    # ksam, kinit = jrnd.split(jrnd.PRNGKey(args.seed))
    # dist.initialize_model(kinit, n_chain)

    # print(f"Running {n_chain} chains of NUTS for {n_iter} iter w/ {n_warm} warmup (tuning) iter...")
    # tic1 = pd.Timestamp.now()
    # run_nuts(ksam, dist.logprob_fn, dist.init_params, 
    #     n_warm, n_iter, n_chain)
    # tic2 = pd.Timestamp.now()
    # print("Runtime for NUTS", tic2 - tic1)

    # print(f"Running {n_chain} chains of NeuTra for {n_iter} iter w/ {vi_iter} precond iter and {n_warm} warmup (tuning) iter...")
    # tic1 = pd.Timestamp.now()
    # run_neutra(ksam, dist.logprob_fn, dist.init_params, 
    #     optim, N_PARAM, n_flow, n_hidden, args.non_linearity, 
    #     n_atoms, vi_iter, n_warm, n_iter, n_chain, nuts=True)
    # tic2 = pd.Timestamp.now()
    # print("Runtime for NeuTra", tic2 - tic1)

    # print(f"Running {n_chain} chains of ATESS for {n_iter} iter w/ {vi_iter} precond iter, {n_epochs} warmup iter of {batch_size} each...")
    # tic1 = pd.Timestamp.now()
    # run_atess(ksam, dist.logprob_fn, dist.init_params, 
    #     optim, N_PARAM, n_flow, n_hidden, args.non_linearity, 
    #     n_atoms, vi_iter, n_epochs, batch_size, n_iter, n_chain)
    # tic2 = pd.Timestamp.now()
    # print("Runtime for ATESS", tic2 - tic1)

    # # pop_pred = Predictive(dist.model, mcmc.get_samples())(jrnd.PRNGKey(2))["y"]
    # # mu = jnp.mean(pop_pred, 0)
    # # pi = jnp.percentile(pop_pred, jnp.array([10, 90]), 0)
    # # plt.figure(figsize=(8, 6), constrained_layout=True)
    # # plt.plot(data.year, data.prey, "ko", mfc="none", ms=4, label="true hare", alpha=0.67)
    # # plt.plot(data.year, data.pred, "bx", label="true lynx")
    # # plt.plot(data.year, mu[:, 0], "k-.", label="pred hare", lw=1, alpha=0.67)
    # # plt.plot(data.year, mu[:, 1], "b--", label="pred lynx")
    # # plt.fill_between(data.year, pi[0, :, 0], pi[1, :, 0], color="k", alpha=0.2)
    # # plt.fill_between(data.year, pi[0, :, 1], pi[1, :, 1], color="b", alpha=0.3)
    # # plt.gca().set(ylim=(0, 160), xlabel="year", ylabel="population (in thousands)")
    # # plt.title("Posterior predictive (80% CI) with predator-prey pattern.")
    # # plt.legend()

    # # plt.savefig("ode_plot_truncnorm.pdf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=0
    )
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-b', '--batch', type=int, default=10_000)
    parser.add_argument('-na', '--n_atoms', type=int, default=4096)
    parser.add_argument(
        "-mu", "--prior-means", type=float, nargs='+',
        help="Prior means for [alpha, beta, gamma, delta, (log)init_predator, (log)init_prey, (log)sd_predator, (log)sd_prey]",
        default=[1., .05, 1., .05, jnp.log(10), jnp.log(10), -1., -1.]
    )
    parser.add_argument(
        "-sd", "--prior-scale", type=float, nargs='+',
        help="Prior scales for [alpha, beta, gamma, delta, (log)init_predator, (log)init_prey, (log)sd_predator, (log)sd_prey]",
        default=[.5, .05, .5, .05, 1., 1., 1., 1.]
    )
    parser.add_argument(
        "-t", "--transport-param", type=int, nargs=4,
        help="Transport parameters [hidden_size, n_hidden_layers, n_flow, n_atoms]",
        default=[N_PARAM, 2, 2, 10]
    )
    parser.add_argument(
        '-lr', '--learning-rate', type=float, default=1e-3
    )
    parser.add_argument(
        "--non-linearity", type=Tuple,
        help="Non-linearity used for training transport.", default=stax.Tanh
    )
    parser.add_argument(
        "--optimizer", type=optax.GradientTransformation,
        help="Optimizer used for training transport.", default=optax.adam
    )
    parser.add_argument(
        "-w", "--warmup-param", type=int, nargs=3,
        help="Warm up parameters [n_iter, n_epochs, batch_size]",
        default=[1000, 5, 1000]
    )
    parser.add_argument(
        "-s", "--sampling-param", type=int, nargs=2,
        help="Sampling parameters [n_iter, n_chain]",
        default=[1000, 4]
    )
    args = parser.parse_args()
    main(args)