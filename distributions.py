import abc

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm, gamma, bernoulli, t
from jax.scipy.special import expit

import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model

import diffrax


class Distribution(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def logprob(self, x1, x2):
        """defines the log probability function"""

    def logprob_fn(self, x):
        return self.logprob(**x)

    @abc.abstractmethod
    def initialize_model(self, rng_key, n_chain):
        """defines the initialization of paramters"""



class RegimeSwitchHMM:

    def __init__(self, T, y) -> None:
        self.T = T
        self.y = y

    def model(self, y=None):
        rho = numpyro.sample('rho', dist.TruncatedNormal(1., .1, low=0.))
        alpha = numpyro.sample('alpha', dist.Normal(0., .1).expand([2]))
        sigma = numpyro.sample('sigma', dist.HalfCauchy(1.).expand([2]))
        p = numpyro.sample('p', dist.Beta(10., 2.).expand([2]))
        xi_0 = numpyro.sample('xi_0', dist.Beta(2., 2.))
        y_0 = numpyro.sample('y_0', dist.Normal(0., 1.))

        numpyro.sample('obs', RegimeMixtureDistribution(
            alpha, rho, sigma, p, xi_0, y_0, self.T
        ), obs=y)

    def initialize_model(self, rng_key, n_chain):
    
        (init_params, *_), self.potential_fn, *_ = initialize_model(
            rng_key, self.model, model_kwargs={'y': self.y},
            dynamic_args=True,
        )
        kchain = jax.random.split(rng_key, n_chain)
        flat, unravel_fn = jax.flatten_util.ravel_pytree(init_params)
        self.init_params = jax.vmap(lambda k: unravel_fn(jax.random.normal(k, flat.shape)))(kchain)
        # self.init_params = jax.vmap(lambda k: unravel_fn(flat))(kchain)

    def logprob_fn(self, params):
        return -self.potential_fn(self.y)(params)

class RegimeMixtureDistribution(dist.Distribution):
    arg_constraints = {
        'alpha': dist.constraints.real,
        'rho': dist.constraints.positive,
        'sigma': dist.constraints.positive,
        'p': dist.constraints.interval(0, 1),
        'xi_0': dist.constraints.interval(0, 1),
        'y_0': dist.constraints.real,
        'T': dist.constraints.positive_integer,
    }
    support = dist.constraints.real
    # reparametrized_params = [] #for VI

    def __init__(self, 
        alpha, rho, sigma, p, xi_0, y_0, T,
        validate_args=True
    ):
        self.alpha, self.rho, self.sigma, self.p, self.xi_0, self.y_0, self.T = (
            alpha, rho, sigma, p, xi_0, y_0, T
        )
        super().__init__(event_shape=(T,), validate_args=validate_args)
        # super().__init__(batch_shape=(T,), validate_args=validate_args)

    def log_prob(self, value):
        def obs_t(carry, y):
            y_prev, xi_1 = carry
            eta_1 = norm.pdf(y, loc=self.alpha[0], scale=self.sigma[0])
            eta_2 = norm.pdf(y, loc=self.alpha[1] + y_prev * self.rho, scale=self.sigma[1])
            lik_1 = self.p[0] * eta_1 + (1 - self.p[0]) * eta_2
            lik_2 = (1 - self.p[1]) * eta_1 + self.p[1] * eta_2
            lik = xi_1 * lik_1 + (1 - xi_1) * lik_2
            lik = jnp.clip(lik, a_min=1e-6)
            return (y, xi_1 * lik_1 / lik), jnp.log(lik)
        _, log_liks = jax.lax.scan(obs_t, (self.y_0, self.xi_0), value)
        return jnp.sum(log_liks)

    def sample(self, key, sample_shape=()):
        return jnp.zeros(sample_shape + self.event_shape)


class HorseshoeLogisticReg(Distribution):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def initialize_model(self, rng_key, n_chain):
        kb, kl, kt = jax.random.split(rng_key, 3)
        self.init_params = {
            'beta': jax.random.normal(kb, (n_chain, self.X.shape[1])),
            'lamda': jax.random.normal(kl, (n_chain, self.X.shape[1])),
            'tau': jax.random.normal(kt, (n_chain,)),
        }

    def logprob(self, beta, lamda, tau): #non-centered
        #priors
        lprob = jnp.sum(
            norm.logpdf(beta, loc=0., scale=1.) +
            gamma.logpdf(jnp.exp(lamda), a=.5, loc=0., scale=2.) + lamda
        ) + gamma.logpdf(jnp.exp(tau), a=.5, loc=0., scale=2.) + tau
        #likelihood
        logit = jnp.sum(self.X * (jnp.exp(tau) * beta * jnp.exp(lamda)), axis=1)
        p = jnp.clip(expit(logit), a_min=1e-6, a_max=1-1e-6)
        # p = jsp.special.expit(logit)
        lprob += jnp.sum(bernoulli.logpmf(self.y, p))
        return lprob


class ProbitReg(Distribution):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def logprob(self, beta):
        lprob = jnp.sum(norm.logpdf(beta, loc=1., scale=1.))
        p = norm.cdf(self.X @ beta)
        p = jnp.clip(p, a_min=1e-6, a_max=1-1e-6)
        lprob += jnp.sum(bernoulli.logpmf(self.y, p))
        return lprob

    def initialize_model(self, rng_key, n_chain):
        self.init_params = {
            'beta': jax.random.normal(rng_key, (n_chain, self.X.shape[1])),
        }


class PredatorPrey:
    def __init__(self, 
        time,
        pred_data, prey_data,
    ) -> None:
        self.time = time
        self.data = jnp.stack([prey_data, pred_data]).T

    def model(self, y=None):

        z_init = numpyro.sample("z_init", dist.LogNormal(jnp.log(10), 1).expand([2]))
        theta = numpyro.sample(
            "theta",
            dist.TruncatedNormal(
                low=0.0,
                loc=jnp.array([1.0, 0.05, 1.0, 0.05]),
                scale=jnp.array([0.5, 0.05, 0.5, 0.05]),
            ),
        )

        term = diffrax.ODETerm(dz_dt)
        # solver = diffrax.Dopri5()
        # solver = diffrax.Tsit5()
        # solver = diffrax.Dopri8()
        solver = diffrax.Heun()
        z = diffrax.diffeqsolve(
            term, solver, 
            t0=0, t1=self.time.shape[0], dt0=1, 
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-5),
            y0=z_init, args=theta,
            max_steps=1000,
            throw=False,
        ).ys
        z = jnp.clip(z, a_min=1e-6)

        sigma = numpyro.sample("sigma", dist.LogNormal(-1, 1).expand([2]))
        numpyro.sample("y", dist.LogNormal(jnp.log(z), sigma), obs=y)


    def initialize_model(self, rng_key, n_chain):
    
        (init_params, *_), self.potential_fn, *_ = initialize_model(
            rng_key, self.model, model_kwargs={'y': self.data},
        )
        kchain = jax.random.split(rng_key, n_chain)
        flat, unravel_fn = jax.flatten_util.ravel_pytree(init_params)
        self.init_params = jax.vmap(lambda k: unravel_fn(jax.random.normal(k, flat.shape)))(kchain)

    def logprob_fn(self, params):
        return -self.potential_fn(params)

def dz_dt(t, z, theta):
    """
    Lotka–Volterra equations. Real positive parameters `alpha`, `beta`, `gamma`, `delta`
    describes the interaction of two species.
    """
    u = z[0]
    v = z[1]
    alpha, beta, gamma, delta = (
        theta[..., 0],
        theta[..., 1],
        theta[..., 2],
        theta[..., 3],
    )
    du_dt = (alpha - beta * v) * u
    dv_dt = (-gamma + delta * u) * v
    return jnp.stack([du_dt, dv_dt])


# class BiDistribution(metaclass=abc.ABCMeta):
class BiDistribution(Distribution):

    # @abc.abstractmethod
    # def logprob(self, x1, x2):
    #     """defines the log probability function"""

    # def logprob_fn(self, x):
    #     return self.logprob(**x)

    def initialize_model(self, rng_key, n_chain):
        ki1, ki2 = jax.random.split(rng_key)
        self.init_params = {
            'x1': jax.random.normal(ki1, shape=(n_chain,)), 
            'x2': jax.random.normal(ki2, shape=(n_chain,))
        }

class BioOxygen(Distribution):

    def __init__(self, times, obs, var) -> None:
        self._times = times
        self._obs = obs
        self._var = var
        super().__init__()

    def initialize_model(self, rng_key, n_chain):
        ki1, ki2 = jax.random.split(rng_key)
        self.init_params = {
            'x1': jax.random.normal(ki1, shape=(n_chain,)), 
            'x2': jax.random.normal(ki2, shape=(n_chain,))
        }

    def logprob(self, x1, x2):
        return -.5 / self._var * jnp.sum((x1 * (1. - jnp.exp(-x2 * self._times)) - self._obs) ** 2)


class Banana(BiDistribution):
    def logprob(self, x1, x2):
        # return norm.logpdf(x1, 0.0, jnp.sqrt(8.0)) + norm.logpdf(
        #     x2, 1 / 4 * x1**2, 1.0
        # )
        return norm.logpdf(x1, 1.0, 1.0) + norm.logpdf(x2, -1.0, 1.0)

# class NealsFunnel(BiDistribution):
class NealsFunnel(Distribution):

    def __init__(self, d=2):
        super().__init__()
        self._d = d

    def logprob(self, x1, x2):
        return norm.logpdf(x1, 0.0, 1.) + jnp.sum(norm.logpdf(
            x2, 0., jnp.exp(2. * x1)
        ))

    def initialize_model(self, rng_key, n_chain):
        ki1, ki2 = jax.random.split(rng_key)
        self.init_params = {
            'x1': jax.random.normal(ki1, shape=(n_chain,)), 
            'x2': jax.random.normal(ki2, shape=(n_chain, self._d-1))
        }

class StudentT(BiDistribution):
    def __init__(self, df=5.) -> None:
        super().__init__()
        self._df = df
    
    def logprob(self, x1, x2):
        return t.logpdf(x1, self._df) + t.logpdf(x2, self._df)

class MixtureNormal(BiDistribution):
    def __init__(self, w1=.2, w2=.8) -> None:
        super().__init__()
        self._w1 = w1
        self._w2 = w2

    def logprob(self, x1, x2):
        return jnp.log(
            self._w1 * norm.pdf(x1, 1., .5) * norm.pdf(x2, 1., .5)
            + self._w2 * norm.pdf(x1, -2., .1) * norm.pdf(x2, -2., .1)
        )

