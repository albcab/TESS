import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.example_libraries import stax
from jax.flatten_util import ravel_pytree

from jax.experimental.host_callback import id_print

from numpyro.diagnostics import print_summary

from kernels import atransp_elliptical_slice, elliptical_slice
from mcmc_utils import inference_loop, inference_loop0
from nn_utils import affine_iaf_masks, MaskedDense, optimize

import blackjax
import jaxopt

from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from numpyro.infer.reparam import NeuTraReparam
from numpyro.infer.autoguide import AutoIAFNormal, AutoDiagonalNormal
from numpyro.infer.initialization import init_to_feasible


### ATESS no transformation

def run_ess(
    rng_key, logprob_fn, init_params,
    n, n_warm, n_iter, n_chain,
):
    ess = elliptical_slice(logprob_fn, n)
    def one_chain(ksam, init_x):
        kwarm, ksam = jrnd.split(ksam)
        state = ess.init(init_x)
        states, info = inference_loop0(kwarm, state, ess.step, n_warm)
        state = jax.tree_map(lambda x: x[-1], states)
        states, info = inference_loop0(ksam, state, ess.step, n_iter)
        return states.position, info.subiter.mean()
    ksam = jrnd.split(rng_key, n_chain)
    samples, subiter = jax.vmap(one_chain)(ksam, init_params)
    print_summary(samples)
    return samples


### ATESS

def run_atess(
    rng_key, logprob_fn, init_params,
    optim, n, n_flow, n_hidden, non_linearity,
    n_atoms, vi_iter, n_epochs, batch_size,
    n_iter, n_chain,
):
    atess, warm_fn = atransp_elliptical_slice(logprob_fn, optim, n, n_flow, n_hidden, non_linearity)
    def one_chain(ksam, init_x):
        kinit, kwarm, ksam = jrnd.split(ksam, 3)
        state, param, err = atess.init(kinit, init_x)#, n_atoms, vi_iter)
        id_print(jnp.zeros(1))
        (state, param), error = warm_fn(kwarm, state, param, n_epochs, batch_size)#, vi_iter)
        id_print(jnp.ones(1))
        states, info = inference_loop(ksam, state, atess.step, n_iter, param)
        return states.position, info.subiter.mean(), (err, param, error)
    ksam = jrnd.split(rng_key, n_chain)
    samples, subiter, diagnose = jax.vmap(one_chain)(ksam, init_params)
    print_summary(samples)
    return samples, diagnose

### NeuTra

def run_neutra(
    # rng_key, data, model, init_params,
    rng_key, logprob_fn, init_params,
    optim, n, n_flow, n_hidden, non_linearity,
    n_atoms, pre_iter, n_warm,
    n_iter, n_chain, nuts=False,
):
    # guide = AutoIAFNormal(model, num_flows=n_flow, hidden_dims=n_hidden, nonlinearity=non_linearity)
    # # guide = AutoDiagonalNormal(model)
    # svi = SVI(model, guide, optim, loss=Trace_ELBO(n_atoms))

    # ksam, kmcmc = jrnd.split(rng_key)
    # params = svi.run(ksam, pre_iter, data, stable_update=True, progress_bar=False).params
    # # id_print(params)
    # neutra = NeuTraReparam(guide, params)
    # reparam_model = neutra.reparam(model)

    # mcmc = MCMC(NUTS(reparam_model), num_warmup=n_warm, num_samples=n_iter, num_chains=n_chain, progress_bar=False)
    # mcmc.run(kmcmc, data, init_params=init_params)
    # mcmc.print_summary(exclude_deterministic=False)
    # return mcmc
    masks = affine_iaf_masks(n, len(n_hidden))
    layers = []
    for mask in masks[:-1]:
        layers.append(MaskedDense(mask))
        layers.append(non_linearity)
    layers.append(MaskedDense(masks[-1]))
    param_init_, Psi_ = stax.serial(*layers)
    def param_init(key, shape):
        keys = jrnd.split(key, n_flow)
        return None, jax.vmap(lambda k: param_init_(k, shape)[1])(keys)
    Psi = lambda param, u: Psi_(param, u).reshape(2, -1)

    def T(u, param):
        u, unravel_fn = ravel_pytree(u)
        def flow_iter(carry, param):
            u, ldj = carry
            psi = Psi(param, u)
            x = jnp.exp(psi[1]) * u + psi[0]
            x = jax.tree_util.tree_map(lambda x: x[::-1], x)
            ldj += jnp.sum(psi[1])
            return (x, ldj), None
        (x, ldj), _ = jax.lax.scan(flow_iter, (u, 0.), param)
        x = jax.tree_util.tree_map(lambda x: x[::-1], x)
        return unravel_fn(x), ldj

    def pi_tilde(param, u):
        x, ldj = T(u, param)
        return -logprob_fn(x) - ldj
    kld = lambda param, U: jnp.sum(jax.vmap(pi_tilde, (None, 0))(param, U))

    # solver = jaxopt.OptaxSolver(kld, optim, maxiter=n_iter, tol=1e-2)
    _, unraveler_fn = ravel_pytree(jax.tree_util.tree_map(lambda x: x[-1], init_params))
    ks, ku, kp = jrnd.split(rng_key, 3)

    # U = jax.vmap(unraveler_fn)(jrnd.normal(ku, shape=(n_atoms, n)))
    ITER = 5
    U = jax.vmap(jax.vmap(unraveler_fn))(jrnd.normal(ku, shape=(ITER, int(n_atoms / ITER), n)))
    pre_iter = int(pre_iter / ITER)

    init_param = param_init(kp, (n,))[1]
    # param, err = optimize(init_param, solver, pre_iter, U)
    param, err = optimize(init_param, kld, optim, pre_iter, U)
    # id_print(err)
    
    pullback_fn = lambda u: -pi_tilde(param, u)
    if nuts:
        samples = run_nuts(ks, pullback_fn, init_params, n_warm, n_iter, n_chain)
    else:
        samples = run_hmc(ks, pullback_fn, init_params, n_warm, n_iter, n_chain)
    samples = jax.vmap(jax.vmap(lambda u: T(u, param)[0]))(samples)
    print_summary(samples)
    return samples
    

### NUTS

def run_nuts(
    # rng_key, data, model, init_params,
    rng_key, logprob_fn, init_params,
    n_warm, n_iter, n_chain,
):
    # mcmc = MCMC(NUTS(model), num_warmup=n_warm, num_samples=n_iter, num_chains=n_chain, progress_bar=False)
    # mcmc.run(rng_key, data, init_params=init_params)
    # mcmc.print_summary(exclude_deterministic=False)
    # return mcmc
    def one_chain(ksam, init_param):
        kwarm, ksam = jrnd.split(ksam)
        # id_print(jnp.zeros(1))
        state, kernel = run_hmc_warmup(kwarm, logprob_fn, init_param, n_warm, .8, True, nuts=True)
        # id_print(jnp.zeros(1)+1)
        states, info = inference_loop0(ksam, state, kernel, n_iter)
        # id_print(jnp.zeros(1)+2)
        return states.position, info
    ksam = jrnd.split(rng_key, n_chain)
    samples, info = jax.vmap(one_chain)(ksam, init_params)
    print_summary(samples)
    return samples

### HMC

def run_hmc(
    rng_key, logporb_fn, init_params,
    n_warm, n_iter, n_chain,
):
    def one_chain(ksam, init_param):
        kwarm, ksam = jrnd.split(ksam)
        state, kernel = run_hmc_warmup(kwarm, logporb_fn, init_param, n_warm, .65, True, nuts=False)
        states, info = inference_loop0(ksam, state, kernel, n_iter)
        return states.position, info
    ksam = jrnd.split(rng_key, n_chain)
    samples, info = jax.vmap(one_chain)(ksam, init_params)
    print_summary(samples)
    return samples

### HMC Tuning

def run_hmc_warmup(
    rng_key, logprob_fn, init_param,
    n_warm, target_acceptance_rate, is_mass_matrix_diagonal,
    nuts=False, progress_bar=False
):
    warmup = blackjax.window_adaptation(
        algorithm=blackjax.nuts if nuts else blackjax.hmc,
        logprob_fn=logprob_fn,
        num_steps=n_warm,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        target_acceptance_rate=target_acceptance_rate,
        progress_bar=progress_bar,
    )
    state, kernel, _ = warmup.run(rng_key, init_param)
    return state, kernel
