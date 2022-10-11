import warnings

import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.flatten_util import ravel_pytree

import pandas as pd
from scipy.stats import qmc

from numpyro.diagnostics import print_summary#, effective_sample_size

from flows import Coupling, ShiftScale
from distances import kullback_liebler, renyi_alpha
from mcmc_utils import inference_loop0, stein_disc, autocorrelation

import blackjax


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

def find_init_step_size(key, logprob_fn, n_param, init_position, batch_fn):
    def one_position(position):
        state = blackjax.hmc.init(position, logprob_fn)
        def while_fn(step_acc):
            step_size, _ = step_acc
            step_size /= 2.
            _, info = blackjax.hmc.kernel()(key, state, logprob_fn, step_size, jnp.ones(n_param), 1)
            acceptance = info.acceptance_probability
            return (step_size, acceptance)
        step_size, _ = jax.lax.while_loop(
            lambda sa: sa[1] < 0.8,
            while_fn, (2., 0.)
        )
        return step_size
    step_size = batch_fn(one_position)(init_position).mean()
    print("Initial step size=", step_size)
    return step_size

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

    print("MEADS")
    samples1 = run_meads(ksam, dist.logprob_fn, batch_init_params,
        n_warm, n_iter, batch_iter, batch_size, batch_fn)

    print("TESS w/ precond.")
    samples, param = run_tess(ksam, dist.logprob_fn, dist.init_params,
        n_warm, n_iter, precond_param, optim, flow, forward, 
        batch_iter, batch_size, args.max_iter, batch_fn)

    halton_sampler = qmc.Halton(d=1, scramble=True, seed=args.seed)
    halton_sample = halton_sampler.random(n=n_warm+n_iter).squeeze()

    init_step_size = args.init_step_size or find_init_step_size(ksam, dist.logprob_fn, N_PARAM, push_init_params, batch_fn)

    print("ChESS-HMC")
    run_chess(ksam, dist.logprob_fn, push_init_params,
        n_warm, n_iter, optim, batch_iter, batch_size, init_step_size, halton_sample, batch_fn)

    print("NUTS w/ adapt.")
    run_nuts(ksam, dist.logprob_fn, push_init_params,
        n_warm, n_iter, batch_iter, batch_size, init_step_size, batch_fn)

    def precond_logprob_fn(u):
        x, ldj = flow(u, precond_param)
        return dist.logprob_fn(x) + ldj
    batch_flow = batch_fn(batch_fn(lambda u: flow(u, precond_param)[0]))

    batch_init_params = jax.tree_map(lambda p: p.reshape((batch_iter, batch_size) + p.shape[1:]), dist.init_params)

    print("MEADS w/ fixed precond.")
    samples = run_meads(ksam, precond_logprob_fn, batch_init_params,
        n_warm, n_iter, batch_iter, batch_size, batch_fn)

    samples = batch_flow(samples)
    do_summary(samples, dist.logprob_fn, 1.)
    
    init_step_size = args.p_init_step_size or find_init_step_size(ksam, precond_logprob_fn, N_PARAM, dist.init_params, batch_fn)

    print("ChESS-HMC w/ fixed precond.")
    samples = run_chess(ksam, precond_logprob_fn, dist.init_params,
        n_warm, n_iter, optim, batch_iter, batch_size, init_step_size, halton_sample, batch_fn)

    samples = batch_flow(samples)
    do_summary(samples, dist.logprob_fn, 1.)

    print("NUTS w/ adapt w/ fixed precond.")
    samples = run_nuts(ksam, precond_logprob_fn, dist.init_params,
        n_warm, n_iter, batch_iter, batch_size, init_step_size, batch_fn)

    samples = batch_flow(samples)
    do_summary(samples, dist.logprob_fn, 1.)

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


def run_chess(
    rng_key, logprob_fn,
    init_position, n_warm, n_iter,
    optim, batch_iter, batch_size,
    init_step_size, halton_sequence,
    batch_fn = jax.pmap,
):
    check_shapes = jax.tree_leaves(
        jax.tree_map(lambda p: p.shape[0] == batch_iter * batch_size, init_position)
    )
    if not all(check_shapes):
        raise ValueError("Num. of chains on initial positions don't match batch_size * batch_iter")

    tic1 = pd.Timestamp.now()
    k_warm, k_sample = jrnd.split(rng_key)
    # init_step_size = .01 # google_stock
    # init_step_size = .1 # predator-prey
    # init_step_size = .0000001 # biooxygen
    warmup = blackjax.chess(logprob_fn, optim, batch_iter * batch_size, halton_sequence, n_warm, init_step_size, batch_fn=batch_fn)
    chain_state, kernel, _ = warmup.run(k_warm, init_position)
    init_state = chain_state.states
    def one_chain(k_sam, init_state):
        state, info = inference_loop0(k_sam, init_state, kernel, n_iter)
        return state.state.position, info
    k_sample = jrnd.split(k_sample, batch_iter * batch_size)
    samples, infos = batch_fn(one_chain)(k_sample, init_state)
    tic2 = pd.Timestamp.now()

    sec = (tic2 - tic1).total_seconds()
    do_summary(samples, logprob_fn, sec)
    print("Runtime for ChESS", (tic2 - tic1).total_seconds())
    return samples


def run_nuts(
    rng_key, logprob_fn,
    init_position, n_warm, n_iter,
    batch_iter, batch_size,
    init_step_size,
    batch_fn = jax.pmap,
):
    check_shapes = jax.tree_leaves(
        jax.tree_map(lambda p: p.shape[0] == batch_iter * batch_size, init_position)
    )
    if not all(check_shapes):
        raise ValueError("Num. of chains on initial positions don't match batch_size * batch_iter")

    tic1 = pd.Timestamp.now()
    k_warm, k_sample = jrnd.split(rng_key)
    warmup = blackjax.nuts_adaptation(logprob_fn, batch_iter * batch_size, init_step_size, n_warm, batch_fn=batch_fn)
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
    print("Runtime for NUTS", (tic2 - tic1).total_seconds())
    return samples