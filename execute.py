import warnings

import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.flatten_util import ravel_pytree

from jax.experimental.host_callback import id_print

import pandas as pd

from numpyro.diagnostics import print_summary#, effective_sample_size

from flows import Coupling, ShiftScale
from distances import kullback_liebler, renyi_alpha
from mcmc_utils import inference_loop0, stein_disc, autocorrelation

import blackjax

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_ts(ax, samples, y):
    c, n = samples['rho'].shape
    t, = y.shape

    def posterior_sample(rng, sample):
        alpha = sample['alpha']
        p = sample['p']
        rho = sample['rho']
        sigma = sample['sigma']
        xi_0 = sample['xi_0']
        y_0 = sample['y_0']
        rngs = jrnd.split(rng, t)
        def obs_t(carry, y_rng):
            y_prev, xi_1 = carry
            y, rng = y_rng
            
            k_s, k_y = jrnd.split(rng, 2)
            std_y = jrnd.normal(k_y)
            y1 = alpha[0] + sigma[0] * std_y
            y2 = alpha[1] + rho * y_prev + sigma[1] * std_y
            y_pred = jnp.where(jrnd.bernoulli(k_s, xi_1), y1, y2)

            eta_1 = jax.scipy.stats.norm.pdf(y, loc=alpha[0], scale=sigma[0])
            eta_2 = jax.scipy.stats.norm.pdf(y, loc=alpha[1] + y_prev * rho, scale=sigma[1])
            lik_1 = p[0] * eta_1 + (1 - p[0]) * eta_2
            lik_2 = (1 - p[1]) * eta_1 + p[1] * eta_2
            lik = xi_1 * lik_1 + (1 - xi_1) * lik_2
            lik = jnp.clip(lik, a_min=1e-6)
            return (y, xi_1 * lik_1 / lik), y_pred
        _, y_pred = jax.lax.scan(obs_t, (y_0, xi_0), (y, rngs))
        return y_pred
    rng = jrnd.split(jrnd.PRNGKey(0), c)
    rngs = jax.vmap(lambda r: jrnd.split(r, n))(rng)
    y_pred = jax.vmap(jax.vmap(posterior_sample))(rngs, samples).reshape(-1, t)
    y_mean = jnp.mean(y_pred, axis=0)
    ts = jnp.arange(1, t+1)
    ax.plot(ts, y, color='r')
    ax.plot(ts, y_mean, color='k')
    for y_p in y_pred:
        ax.plot(ts, y_p, color='k', alpha=.005)
    return None

def plot_density(samples, post=''):
    samples = jax.tree_map(lambda s: s.reshape((-1,) + s.shape[2:]), samples)
    sam = []
    names = []
    for name, value in samples.items():
        sam.append(value.T)
    samples = jnp.vstack(sam).T
    df = pd.DataFrame(samples)
    df.columns = [r'$\alpha$', r'$\beta$', r'$\delta$', r'$\gamma$', r'$p(0)$', r'$q(0)$', r'$\sigma_p$', r'$\sigma_q$']
    # df.columns = [r'$\alpha_1$', r'$\alpha_2$', r'$p_{1,1}$', r'$p_{2,2}$', r'$\rho$', r'$\sigma_1$', r'$\sigma_2$', r'$\xi_{10}$', r'$r_0$']
    sns.pairplot(df, kind='kde', diag_kind='kde')#, plot_kws={'bins': 50})
    plt.savefig('hist'+post+'.png')
    plt.close()
    return None

def plots(samples, param, flow, flow_inv, sam, sam2):

    np.random.seed(0)
    c, n = samples["x1"].shape
    u1 = np.random.normal(0., 1., size=c * n)
    u2 = np.random.normal(0., 1., size=c * n)
    x1 = samples["x1"].reshape(-1)
    x2 = samples["x2"].reshape(-1)
    phi_samples, phi_weights = jax.vmap(lambda u1, u2: flow(jnp.array([u1, u2]), param))(u1, u2)
    pi_samples, pi_weights = jax.vmap(lambda x1, x2: flow_inv(jnp.array([x1, x2]), param))(x1, x2)
    w = jnp.exp(phi_weights)
    print(jnp.min(w), jnp.max(w))
    w = jnp.exp(pi_weights)
    print(jnp.min(w), jnp.max(w))

    fig, ax = plt.subplots(1, 4, figsize=(22, 4))#, sharex=True, sharey=True)
    ax[3].set_title(r"$\phi(u)$")
    ax[3].set_xlabel(r"$u_1$")
    ax[3].set_ylabel(r"$u_2$")
    # sns.kdeplot(x=u1, y=u2, ax=ax[3], fill=True)
    sns.histplot(x=u1, y=u2, ax=ax[3], bins=50)
    ax[0].set_title(r"$\hat{\pi}(u)$")
    ax[0].set_xlabel(r"$u_1$")
    ax[0].set_ylabel(r"$u_2$")
    # sns.kdeplot(x=pi_samples[:, 0], y=pi_samples[:, 1], ax=ax[0], fill=True)
    sns.histplot(x=pi_samples[:, 0], y=pi_samples[:, 1], weights=jnp.exp(pi_weights), ax=ax[0], bins=50)
    ax[2].set_title(r"$\hat{\phi}(\theta)$")
    ax[2].set_xlabel(r"$\theta_1$")
    ax[2].set_ylabel(r"$\theta_2$")
    # sns.kdeplot(x=phi_samples[:, 0], y=phi_samples[:, 1], ax=ax[2], fill=True)
    sns.histplot(x=phi_samples[:, 0], y=phi_samples[:, 1], weights=jnp.exp(phi_weights), ax=ax[2], bins=50)
    ax[1].set_title(r"$\pi(\theta)$")
    ax[1].set_xlabel(r"$\theta_1$")
    ax[1].set_ylabel(r"$\theta_2$")
    # sns.kdeplot(x=x1, y=x2, ax=ax[1], fill=True)
    sns.histplot(x=x1, y=x2, ax=ax[1], bins=50)
    ax[1].sharex(ax[2])
    ax[1].sharey(ax[2])
    ax[0].sharex(ax[3])
    ax[0].sharey(ax[3])
    plt.savefig("biooxygen.png", bbox_inches='tight')
    plt.close()

    # @jax.vmap
    # def mse_iter(x1, x2): 
    #     theta1 = (1. - jnp.cumsum(x1)/jnp.arange(1, n+1))**2
    #     theta2 = (.1 - jnp.cumsum(x2)/jnp.arange(1, n+1))**2
    #     return theta1 + theta2
    # def plot_samples(ax, samples):
    #     mses = mse_iter(samples['x1'], samples['x2'])
    #     mean = jnp.mean(mses, axis=0)
    #     sd = jnp.std(mses, axis=0)
    #     iters = jnp.arange(1, n+1)
    #     ax.plot(iters, mean)
    #     ax.fill_between(iters, mean - 2*sd, mean + 2*sd, alpha=0.2)
    #     return None
    # fig, ax = plt.subplots()
    # ax.set_title("MSE Biochemical oxygen demand model")
    # ax.set_xlabel("Iteration")
    # ax.set_ylabel("MSE")
    # plot_samples(ax, samples)
    # plot_samples(ax, sam)
    # plot_samples(ax, sam2)
    # plt.savefig("biooxygen_mse.png", bbox_inches='tight')

    return None

def do_summary(samples, logprob_fn, sec):
    print_summary(samples)
    stein = stein_disc(samples, logprob_fn)
    print(f"Stein U-, V-statistics={stein[0]}, {stein[1]}")
    # stein = [0, 0]
    corr = []
    ess = []
    for name, value in samples.items():
        value = jax.device_get(value)
        n = value.shape
        auto_corr = autocorrelation(value, axis=1)
        factor = 1. - jnp.arange(1, n[1]) / n[1]
        if len(n) == 3:
            auto_corr = jax.vmap(lambda ac: 1./2 + 2 * jnp.sum(factor * ac[:, 1:], axis=1), 2)(auto_corr)
        else:
            auto_corr = 1./2 + 2 * jnp.sum(factor * auto_corr[:, 1:], axis=1)
        corr.append(auto_corr)
        # ind_ess = effective_sample_size(value)
        # ess.append(ind_ess)
    corr = jnp.vstack(corr).T
    ess = n[1] / (2 * corr)
    ess = jnp.median(ess, axis=0)
    print("Min. ESS=", jnp.min(ess) * n[0], jnp.min(ess))
    # ess = jnp.hstack(ess)
    # print("Min. ESS=", jnp.min(ess), jnp.min(ess)/n[0])
    # corr = jnp.max(corr, axis=0)
    # print("Mean and std max int corr=", jnp.mean(corr), jnp.std(corr))
    std_corr = jnp.std(jnp.max(corr, axis=1))
    corr = jnp.median(corr, axis=0)
    print("Mean and std max int corr=", jnp.max(corr), std_corr)
    print(f"{jnp.max(corr):.3f} & {std_corr:.3f} & " + 
        f"{jnp.min(ess) * n[0]:.0f} & {jnp.min(ess):.0f} & " + 
        f"{jnp.min(ess) * n[0] / sec:.3f} & {jnp.min(ess) / sec:.3f} & " + 
        f"{stein[0]:.3e} & {stein[1]:.3e}")
    return None

def run(dist, args, optim, N_PARAM, batch_fn=jax.vmap):
    [n_warm, n_iter] = args.sampling_param
    [batch_iter, batch_size] = args.batch_shape
    kflow, ksam, kinit = jrnd.split(jrnd.PRNGKey(args.seed), 3)
    dist.initialize_model(kinit, batch_iter * batch_size)

    init_param, flow, flow_inv, reverse, forward = initialize_flow(
        kflow, dist.logprob_fn, args.flow, args.distance, N_PARAM, 
        args.n_flow, args.n_hidden, args.non_linearity, args.num_bins
    )

    one_init_param = jax.tree_map(lambda p: p[0], dist.init_params)
    mc_samples = 1000
    precond_iter = args.preconditon_iter
    precond_param = run_precondition(kflow, init_param, one_init_param, 
        optim, reverse, mc_samples, precond_iter)

    push_init_params = batch_fn(lambda u: flow(u, precond_param)[0])(dist.init_params)
    batch_init_params = jax.tree_map(lambda p: p.reshape((batch_iter, batch_size) + p.shape[1:]), push_init_params)

    # push_init_params = batch_fn(lambda u: u)(dist.init_params)
    # batch_init_params = jax.tree_map(lambda p: p.reshape((batch_iter, batch_size) + p.shape[1:]), push_init_params)

    # print("MEADS")
    # samples1 = run_meads(ksam, dist.logprob_fn, batch_init_params,
    #     n_warm, n_iter, batch_iter, batch_size, batch_fn)

    # plot_density(samples1, '_pp_meads2')

    print("MEDS")
    samples2 = run_meds(ksam, dist.logprob_fn, push_init_params,
        n_warm, n_iter, batch_iter, batch_size, batch_fn)

    # print("TESS no precond.")
    # run_tess(ksam, dist.logprob_fn, dist.init_params,
    #     n_warm, n_iter, init_param, optim, flow, forward, 
    #     batch_iter, batch_size, args.max_iter, batch_fn)

    print("TESS w/ precond.")
    samples, param = run_tess(ksam, dist.logprob_fn, dist.init_params,
        n_warm, n_iter, precond_param, optim, flow, forward, 
        batch_iter, batch_size, args.max_iter, batch_fn)

    plot_density(samples, '_pp2')

    # plots(samples, param, flow, flow_inv, samples1, samples2)

    def precond_logprob_fn(u):
        x, ldj = flow(u, precond_param)
        return dist.logprob_fn(x) + ldj
    batch_flow = batch_fn(batch_fn(lambda u: flow(u, precond_param)[0]))

    simple_init_param, simple_flow, *_, forward = initialize_flow(
        kflow, precond_logprob_fn, 'shift_scale', args.distance, N_PARAM,
        args.n_flow, args.n_hidden, args.non_linearity, args.num_bins
    )

    print("TESS w/ fixed precond. and simple flow")
    samples, _ = run_tess(ksam, precond_logprob_fn, dist.init_params,
        n_warm, n_iter, simple_init_param, optim, simple_flow, forward, 
        batch_iter, batch_size, args.max_iter, batch_fn)

    samples = batch_flow(samples)
    do_summary(samples, dist.logprob_fn, 1.)

    # batch_init_params = jax.tree_map(lambda p: p.reshape((batch_iter, batch_size) + p.shape[1:]), dist.init_params)

    # print("MEADS w/ fixed precond.")
    # samples = run_meads(ksam, precond_logprob_fn, batch_init_params,
    #     n_warm, n_iter, batch_iter, batch_size, batch_fn)

    # samples = batch_flow(samples)
    # do_summary(samples, dist.logprob_fn, 1.)

    # print("MEDS w/ fixed precond.")
    # samples = run_meds(ksam, precond_logprob_fn, dist.init_params,
    #     n_warm, n_iter, batch_iter, batch_size, batch_fn)

    # samples = batch_flow(samples)
    # do_summary(samples, dist.logprob_fn, 1.)

    return None


def full_run(dist, args, optim, N_PARAM, batch_fn=jax.vmap):
    [n_warm, n_iter] = args.sampling_param
    [batch_iter, batch_size] = args.batch_shape
    kflow, ksam, kinit = jrnd.split(jrnd.PRNGKey(args.seed), 3)
    dist.initialize_model(kinit, batch_iter * batch_size)

    init_param, flow, flow_inv, reverse, forward = initialize_flow(
        kflow, dist.logprob_fn, args.flow, args.distance, N_PARAM, 
        args.n_flow, args.n_hidden, args.non_linearity, args.num_bins
    )

    print("TESS no transf. (i.e. no warm)")
    run_tess(ksam, dist.logprob_fn, dist.init_params,
        0, n_iter, init_param, optim, flow, forward, 
        batch_iter, batch_size, args.max_iter, batch_fn)

    batch_init_params = jax.tree_map(lambda p: p.reshape((batch_iter, batch_size) + p.shape[1:]), dist.init_params)
    one_init_param = jax.tree_map(lambda p: p[0], dist.init_params)

    print("MEADS")
    run_meads(ksam, dist.logprob_fn, batch_init_params,
        n_warm, n_iter, batch_iter, batch_size, batch_fn)

    print("MEDS")
    run_meds(ksam, dist.logprob_fn, dist.init_params,
        n_warm, n_iter, batch_iter, batch_size, batch_fn)

    print("ATESS no precond.")
    run_atess(ksam, dist.logprob_fn, batch_init_params,
        n_warm, n_iter, init_param, optim, flow, forward, 
        batch_iter, batch_size, args.max_iter, batch_fn)

    print("TESS no precond.")
    run_tess(ksam, dist.logprob_fn, dist.init_params,
        n_warm, n_iter, init_param, optim, flow, forward, 
        batch_iter, batch_size, args.max_iter, batch_fn)

    mc_samples = 1000
    precond_iter = 1000
    precond_param = run_precondition(kflow, init_param, one_init_param, 
        optim, reverse, mc_samples, precond_iter)

    print("ATESS w/ precond.")
    run_atess(ksam, dist.logprob_fn, batch_init_params,
        n_warm, n_iter, precond_param, optim, flow, forward, 
        batch_iter, batch_size, args.max_iter, batch_fn)

    print("TESS w/ precond.")
    run_tess(ksam, dist.logprob_fn, dist.init_params,
        n_warm, n_iter, precond_param, optim, flow, forward, 
        batch_iter, batch_size, args.max_iter, batch_fn)

    def precond_logprob_fn(u):
        x, ldj = flow(u, precond_param)
        return dist.logprob_fn(x) + ldj
    batch_flow = batch_fn(batch_fn(lambda u: flow(u, precond_param)[0]))

    simple_init_param, simple_flow, *_, forward = initialize_flow(
        kflow, precond_logprob_fn, 'shift_scale', args.distance, N_PARAM,
        args.n_flow, args.n_hidden, args.non_linearity, args.num_bins
    )

    print("TESS no transf. w/ fixed precond.")
    samples = run_tess(ksam, precond_logprob_fn, dist.init_params,
        0, n_iter, simple_init_param, optim, simple_flow, forward, 
        batch_iter, batch_size, args.max_iter, batch_fn)

    samples = batch_flow(samples)
    do_summary(samples, dist.logprob_fn)

    print("ATESS w/ fixed precond. and simple flow")
    samples = run_atess(ksam, precond_logprob_fn, batch_init_params,
        n_warm, n_iter, simple_init_param, optim, simple_flow, forward, 
        batch_iter, batch_size, args.max_iter, batch_fn)

    samples = batch_flow(samples)
    do_summary(samples, dist.logprob_fn)

    print("TESS w/ fixed precond. and simple flow")
    samples = run_tess(ksam, precond_logprob_fn, dist.init_params,
        n_warm, n_iter, simple_init_param, optim, simple_flow, forward, 
        batch_iter, batch_size, args.max_iter, batch_fn)

    samples = batch_flow(samples)
    do_summary(samples, dist.logprob_fn)

    print("MEADS w/ fixed precond.")
    samples = run_meads(ksam, precond_logprob_fn, batch_init_params,
        n_warm, n_iter, batch_iter, batch_size, batch_fn)

    samples = batch_flow(samples)
    do_summary(samples, dist.logprob_fn)

    print("MEDS w/ fixed precond.")
    samples = run_meds(ksam, precond_logprob_fn, dist.init_params,
        n_warm, n_iter, batch_iter, batch_size, batch_fn)

    samples = batch_flow(samples)
    do_summary(samples, dist.logprob_fn)

    return None


non_lins = {
    'tanh': jax.nn.tanh,
    'elu': jax.nn.elu,
    'relu': jax.nn.relu,
    'swish': jax.nn.swish,
}

flows = {
    'coupling': lambda n, f, h, nl, nb: Coupling(n, f, h, nl, False, nb).get_utilities(),
    'ncoupling': lambda n, f, h, nl, nb: Coupling(n, f, h, nl, True, nb).get_utilities(),
    'shift_scale': lambda n, *_: ShiftScale(n).get_utilities(),
}

distances = {
    'kld': kullback_liebler,
    'ralpha=0.5': lambda fn, n, f, fi: renyi_alpha(fn, n, f, fi, .5),
    'ralpha=2': lambda fn, n, f, fi: renyi_alpha(fn, n, f, fi, 2.),
    'ralpha=0': lambda fn, n, f, fi: renyi_alpha(fn, n, f, fi, 0.),
}


def initialize_flow(
    rng_key, logprob_fn, flow, distance, 
    d, n_flow, n_hidden, non_linearity, num_bins,
):
    if flow in ['iaf', 'riaf'] and any([nh != d for nh in n_hidden]):
        warnings.warn('IAF flows always have dimension of hidden units same as params.')

    if flow in ['iaf', 'riaf'] and num_bins:
        warnings.warn('IAF cannot do rational quadratic splines.')
    
    if flow in ['latent', 'nlatent']:
        warnings.warn('NeuTra samples for fully coupled dense flows including latent parameters are irrelevant, see code.')

    print(f"\nTransformation w/ {n_flow} flows (flow: {flow}, splines? {num_bins is not None}) - hidden layers={n_hidden} - {non_linearity} nonlinearity")
    param_init, flow, flow_inv = flows[flow](d, n_flow, n_hidden, non_lins[non_linearity], num_bins)
    reverse, forward = distances[distance](logprob_fn, flow, flow_inv)
    init_param = param_init(rng_key, jrnd.normal(rng_key, shape=(d,)))
    return init_param, flow, flow_inv, reverse, forward


def run_precondition(
    rng_key, init_param, position,
    optim, reverse,
    batch_size, n_iter,
):
    tic1 = pd.Timestamp.now()
    p, unraveler_fn = ravel_pytree(position)
    U = jax.vmap(unraveler_fn)(
        jax.random.normal(rng_key, shape=(batch_size,) + p.shape)
    )
    opt_state = optim.init(init_param)
    (param, opt_state), loss_value = blackjax.adaptation.atess.optimize(
        init_param, opt_state, reverse, 
        optim, n_iter, U,
    )
    tic2 = pd.Timestamp.now()
    print("Runtime for pre-conditioning", tic2 - tic1)
    return param


def run_tess(
    rng_key, logprob_fn,
    init_position, n_warm, n_iter,
    init_param, optim, flow, forward,
    batch_iter, batch_size, maxiter,
    batch_fn = jax.pmap,
):
    check_shapes = jax.tree_leaves(
        jax.tree_map(lambda p: p.shape[0] == batch_iter * batch_size, init_position)
    )
    if not all(check_shapes):
        raise ValueError("Num. of chains on initial positions don't match batch_size * batch_iter")

    tic1 = pd.Timestamp.now()
    k_warm, k_sample = jrnd.split(rng_key)
    if n_warm > 0:
        warmup = blackjax.atess(logprob_fn, optim, init_param, flow, forward, batch_iter, batch_size, n_warm, maxiter, eca=False, batch_fn=batch_fn)
        chain_state, kernel, param = warmup.run(k_warm, init_position)
        init_state = chain_state.states
    else:
        init, kernel = blackjax.tess(logprob_fn, lambda u: (u, 0))
        init_state = batch_fn(init)(init_position)
    def one_chain(k_sam, init_state):
        state, info = inference_loop0(k_sam, init_state, kernel, n_iter)
        return state.position, info.subiter.mean()
    k_sample = jrnd.split(k_sample, batch_iter * batch_size)
    samples, subiter = batch_fn(one_chain)(k_sample, init_state)
    # print(subiter)
    tic2 = pd.Timestamp.now()

    sec = (tic2 - tic1).total_seconds()
    do_summary(samples, logprob_fn, sec)
    print("Runtime for TESS", (tic2 - tic1).total_seconds())
    return samples, param

def run_atess(
    rng_key, logprob_fn,
    init_position, n_warm, n_iter,
    init_param, optim, flow, forward,
    batch_iter, batch_size, maxiter,
    batch_fn = jax.pmap,
):
    check_shapes = jax.tree_leaves(
        jax.tree_map(lambda p: p.shape[:2] == (batch_iter, batch_size), init_position)
    )
    if not all(check_shapes):
        raise ValueError("Num. of chains on initial positions don't match (batch_size, batch_iter)")

    tic1 = pd.Timestamp.now()
    warmup = blackjax.atess(logprob_fn, optim, init_param, flow, forward, batch_iter, batch_size, n_warm+n_iter, maxiter, eca=True, batch_fn=batch_fn)
    _, kernel, states = warmup.run(rng_key, init_position)
    samples = jax.tree_map(lambda s: jnp.swapaxes(s[n_warm:].reshape((n_iter, batch_iter * batch_size) + s.shape[3:]), 0, 1), states.states.position)
    tic2 = pd.Timestamp.now()

    sec = (tic2 - tic1).total_seconds()
    do_summary(samples, logprob_fn, sec)
    print("Runtime for ATESS", (tic2 - tic1).total_seconds())
    return samples    


def run_meds(
    rng_key, logprob_fn,
    init_position, n_warm, n_iter,
    batch_iter, batch_size,
    batch_fn = jax.pmap,
):
    check_shapes = jax.tree_leaves(
        jax.tree_map(lambda p: p.shape[0] == batch_iter * batch_size, init_position)
    )
    if not all(check_shapes):
        raise ValueError("Num. of chains on initial positions don't match batch_size * batch_iter")

    tic1 = pd.Timestamp.now()
    k_warm, k_sample = jrnd.split(rng_key)
    warmup = blackjax.meads(logprob_fn, batch_iter, batch_size, n_warm, eca=False, batch_fn=batch_fn)
    chain_state, kernel, _ = warmup.run(k_warm, init_position)
    init_state = chain_state.states
    def one_chain(k_sam, init_state):
        state, info = inference_loop0(k_sam, init_state, kernel, n_iter)
        return state.position, info
    k_sample = jrnd.split(k_sample, batch_iter * batch_size)
    samples, infos = batch_fn(one_chain)(k_sample, init_state)
    tic2 = pd.Timestamp.now()

    sec = (tic2 - tic1).total_seconds()
    do_summary(samples, logprob_fn, sec)
    print("Runtime for MEDS", (tic2 - tic1).total_seconds())
    return samples

def run_meads(
    rng_key, logprob_fn,
    init_position, n_warm, n_iter,
    batch_iter, batch_size,
    batch_fn = jax.pmap,
):
    check_shapes = jax.tree_leaves(
        jax.tree_map(lambda p: p.shape[:2] == (batch_iter, batch_size), init_position)
    )
    if not all(check_shapes):
        raise ValueError("Num. of chains on initial positions don't match (batch_size, batch_iter)")

    tic1 = pd.Timestamp.now()
    warmup = blackjax.meads(logprob_fn, batch_iter, batch_size, n_warm+n_iter, eca=True, batch_fn=batch_fn)
    _, kernel, states = warmup.run(rng_key, init_position)
    samples = jax.tree_map(lambda s: jnp.swapaxes(s[n_warm:].reshape((n_iter, batch_iter * batch_size) + s.shape[3:]), 0, 1), states.states.position)
    tic2 = pd.Timestamp.now()

    sec = (tic2 - tic1).total_seconds()
    do_summary(samples, logprob_fn, sec)
    print("Runtime for MEADS", (tic2 - tic1).total_seconds())
    return samples
