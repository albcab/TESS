"""Centering flows"""
from typing import Callable

from jax import vmap
import jax.random as jrnd
import jax.numpy as jnp

from numpyro.handlers import seed, replay
from numpyro.infer.util import log_density, init_to_uniform
from numpyro.util import _validate_model, check_model_guide_match
from numpyro.optim import _NumPyroOptim
from numpyro.infer import ELBO
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoContinuous

# from nlls_utils import gauss_newton
from jaxopt import GaussNewton, ProjectedGradient, LBFGS
from jaxopt.projection import projection_box


class AutoStdNormal(AutoContinuous):
    def __init__(
        self,
        model,
        *,
        prefix="auto",
        init_loc_fn=init_to_uniform,
        init_scale=0.1,
    ):
        if init_scale <= 0:
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        super().__init__(model, prefix=prefix, init_loc_fn=init_loc_fn)
        
    def _get_posterior(self):
        loc = jnp.zeros(self.latent_dim)
        scale = jnp.ones(self.latent_dim)
        return dist.Normal(loc, scale)

    def get_base_dist(self):
        return dist.Normal(jnp.zeros(self.latent_dim), 1).to_event(1)

    def get_transform(self, params):
        loc = jnp.zeros(self.latent_dim)
        scale = jnp.ones(self.latent_dim)
        return dist.transforms.IndependentTransform(dist.transforms.AffineTransform(loc, scale), 1)
    

def run_gaussnewton(rng, log_pullback, d, rnrom_size, init_params):
    # rng1, rng2 = jrnd.split(rng)
    # z0 = jnp.log(jnp.abs(jrnd.normal(rng1, (rnrom_size, 1))))
    # Z = jnp.concatenate([z0, jrnd.normal(rng2, (rnrom_size, d-1))], axis=1)
    # lognorm = vmap(lambda z: -.5 * jnp.dot(z[1:], z[1:]) -.5 * jnp.exp(z[0])**2 + z[0])(Z)
    Z = jrnd.normal(rng, (rnrom_size, d))
    lognorm = vmap(lambda z: -d/2. * jnp.log(2*jnp.pi) - 1./2 * jnp.sum(z**2))(Z)
    def residual_fn(params):
        logpulled = vmap(log_pullback, (None, 0))(params, Z)
        T = logpulled - lognorm
        return -jnp.mean(T)
        # res = T - T.mean()
        # return .5 * jnp.dot(res, res)
    # solver = GaussNewton(
    solver = ProjectedGradient(
        residual_fn, 
        projection_box,
        maxiter=1000, tol=0.001, 
        verbose=True, implicit_diff=True,
    )
    params, state = solver.run(
        init_params, 
        hyperparams_proj=(jnp.zeros(d), jnp.ones(d))
    )
    return params

class JaxOptim(_NumPyroOptim):
    def __init__(self, optim_fn: Callable, *args, **kwargs) -> None:
        super().__init__(optim_fn, *args, **kwargs)

    def update(self, params, loss):
        state = self.update_fn(params, loss)
        return 0, state

    def eval_and_update(self, fn, state):
        params = self.get_params(state)
        params, state = self.update(params, fn)
        return (state[1].error, None), (params, state)


def projectiongradient_to_numpyro(hyperparams_proj, **kwargs) -> JaxOptim:
    def init_fn(params):
        return params, 0.

    def update_fn(params, loss_fn):
        loss = lambda params: loss_fn(params)[0]
        solver = ProjectedGradient(loss, projection_box, **kwargs)
        params, opt_state = solver.run(
            params, 
            hyperparams_proj=hyperparams_proj
        )
        return params, opt_state

    def get_params_fn(state):
        params, _ = state
        return params

    return JaxOptim(lambda x, y, z: (x, y, z), init_fn, update_fn, get_params_fn)

def lbfgs_to_numpyro(**kwargs) -> JaxOptim:
    def init_fn(params):
        return params, 0.

    def update_fn(params, loss_fn):
        loss = lambda params: loss_fn(params)[0]
        solver = LBFGS(loss, ** kwargs)
        params, opt_state = solver.run(params)
        return params, opt_state

    def get_params_fn(state):
        params, _ = state
        return params

    return JaxOptim(lambda x, y, z: (x, y, z), init_fn, update_fn, get_params_fn)

def gaussnewton_to_numpyro(**kwargs) -> JaxOptim: #must be used to Trace_NLLS
    def init_fn(params):
        return params, 0.

    def update_fn(params, loss_fn):
        loss = lambda params: loss_fn(params)[0]
        solver = GaussNewton(loss, **kwargs)
        params, opt_state = solver.run(params)
        return params, opt_state

    def get_params_fn(state):
        params, _ = state
        return params

    return JaxOptim(lambda x, y, z: (x, y, z), init_fn, update_fn, get_params_fn)

class Trace_NLLS(ELBO):
    def __init__(self, num_particles):
        self.num_particles = num_particles

    def loss_with_mutable_state(
        self, rng_key, param_map, model, guide, *args, **kwargs
    ):
        def single_particle_elbo(rng_key):
            params = param_map.copy()
            model_seed, guide_seed = jrnd.split(rng_key)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)
            guide_log_density, guide_trace = log_density(
                seeded_guide, args, kwargs, param_map
            )
            mutable_params = {
                name: site["value"]
                for name, site in guide_trace.items()
                if site["type"] == "mutable"
            }
            params.update(mutable_params)
            seeded_model = replay(seeded_model, guide_trace)
            model_log_density, model_trace = log_density(
                seeded_model, args, kwargs, params
            )
            check_model_guide_match(model_trace, guide_trace)
            _validate_model(model_trace, plate_warning="loose")
            mutable_params.update(
                {
                    name: site["value"]
                    for name, site in model_trace.items()
                    if site["type"] == "mutable"
                }
            )

            # log p(z) - log q(z)
            elbo_particle = model_log_density - guide_log_density
            if mutable_params:
                if self.num_particles == 1:
                    return elbo_particle, mutable_params
                else:
                    raise ValueError(
                        "Currently, we only support mutable states with num_particles=1."
                    )
            else:
                return elbo_particle, None

        # Return (-elbo) since by convention we do gradient descent on a loss and
        # the ELBO is a lower bound that needs to be maximized.
        rng_keys = jrnd.split(rng_key, self.num_particles)
        elbos, mutable_state = vmap(single_particle_elbo)(rng_keys)
        return {"loss": elbos - elbos.mean(), "mutable_state": mutable_state}