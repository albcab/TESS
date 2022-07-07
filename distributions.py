import abc

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm, gamma, bernoulli, t
from jax.experimental.ode import odeint
from jax.scipy.special import expit

from jax.experimental.host_callback import id_print

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.util import initialize_model
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer.initialization import init_to_median, init_to_value

import diffrax

from centering import projectiongradient_to_numpyro


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
            # xi_1 = jnp.clip(xi_1, a_min=1e-6, a_max=1-1e-6)
            eta_1 = norm.pdf(y, loc=self.alpha[0], scale=self.sigma[0])
            eta_2 = norm.pdf(y, loc=self.alpha[1] + y_prev * self.rho, scale=self.sigma[1])
            lik_1 = self.p[0] * eta_1 + (1 - self.p[0]) * eta_2
            # lik_1 = jnp.clip(lik_1, a_min=1e-37)
            lik_2 = (1 - self.p[1]) * eta_1 + self.p[1] * eta_2
            # lik_2 = jnp.clip(lik_2, a_min=1e-37)
            lik = xi_1 * lik_1 + (1 - xi_1) * lik_2
            lik = jnp.clip(lik, a_min=1e-6)
            return (y, xi_1 * lik_1 / lik), jnp.log(lik)
        _, log_liks = jax.lax.scan(obs_t, (self.y_0, self.xi_0), value)
        return jnp.sum(log_liks)

    def sample(self, key, sample_shape=()):
        return jnp.zeros(sample_shape + self.event_shape)


class HorseshoeLogisticReg:
    def __init__(
        self, X, y,
        alpha_a = .5, beta_a = .5, 
        alpha_b = .5, beta_b = .5,
    ) -> None:
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.param_tau = jnp.array([alpha_a, alpha_b])
        self.param_lamda = jnp.array([beta_a, beta_b])

    def model(self, y=None): 
        plate_obs = numpyro.plate('i', self.n, dim=-1)
        plate_reg = numpyro.plate('j', self.d, dim=-1)
        
        # tau = numpyro.sample('tau', dist.HalfCauchy(1.))
        # tau = numpyro.sample('tau', dist.LogNormal(0., 1.))
        tau = numpyro.sample('tau', dist.Gamma(*self.param_tau))
        
        with plate_reg:
            # lamda = numpyro.sample('lamda', dist.HalfCauchy(1.))
            # lamda = numpyro.sample('lamda', dist.LogNormal(0., 1.))
            lamda = numpyro.sample('lamda', dist.Gamma(*self.param_lamda))

            beta = numpyro.sample('beta', dist.Normal(0., 1.)) #noncentered
            # beta = numpyro.sample('beta', dist.Normal(0., tau * lamda)) #centered
            
        logit = jnp.sum(self.X * (tau * beta * lamda), axis=1) #noncentered
        # logit = jnp.sum(self.X * beta, axis=1) #centered
        p = jnp.clip(expit(logit), a_min=1e-6, a_max=1-1e-6)
        # p = expit(logit)
        with plate_obs:
            numpyro.sample('obs', dist.Bernoulli(p), obs=y)

    def center_model(self, rng_key, optim, n_atoms, n_iter):
        def model(y=None):
            plate_obs = numpyro.plate('i', self.n, dim=-1)
            plate_reg = numpyro.plate('j', self.d, dim=-1)
            
            tau = numpyro.sample('tau', dist.InverseGamma(*self.param_tau))

            lamda = numpyro.sample('lamda', dist.InverseGamma(*self.param_lamda).expand([self.d]))
            l = numpyro.param('beta_centered', jnp.ones(self.d) * .5, constraint=dist.constraints.interval(0, 1))
            beta = numpyro.sample('beta', dist.Normal(0., (lamda * tau) ** l))
            
            logit = jnp.dot(self.X, (lamda * tau) ** (1-l) * beta)
            # p = expit(logit)
            p = jnp.clip(expit(logit), a_min=1e-6, a_max=1-1e-6)
            # id_print(jnp.min(p))
            with plate_obs:
                numpyro.sample('obs', dist.Bernoulli(p), obs=y)

        # optim = jax.example_libraries.optimizers.adam(lr)
        guide = AutoDiagonalNormal(model, init_loc_fn=init_to_median, init_scale=.1)

        svi = SVI(model, guide, optim, Trace_ELBO(n_atoms))
        params = svi.run(rng_key, n_iter, self.y, progress_bar=False).params
        self.model_autocenter = numpyro.handlers.substitute(model, params)


    def initialize_model(self, rng_key, n_chain, autocenter=False): 

        (init_params, *_), self.potential_fn, *_ = initialize_model(
            rng_key, self.model_autocenter if autocenter else self.model, 
            model_kwargs={'y': self.y},
        )
        kchain = jax.random.split(rng_key, n_chain)
        flat, unravel_fn = jax.flatten_util.ravel_pytree(init_params)
        self.init_params = jax.vmap(lambda k: unravel_fn(jax.random.normal(k, flat.shape)))(kchain)

    def logprob_fn(self, params):
        return -self.potential_fn(params)

        # kb, kl, kt = jax.random.split(rng_key, 3)
        # self.init_param = {
        #     'beta': jax.random.normal(kb, (n_chain, self.X.shape[1])),
        #     'lamda': jax.random.normal(kl, (n_chain, self.X.shape[1])),
        #     'tau': jax.random.normal(kt, (n_chain,)),
        # }

    # def logprob(self, beta, lamda, tau): #non-centered
    #     #priors
    #     lprob = jnp.sum(
    #         norm.logpdf(beta, loc=0., scale=1.) +
    #         gamma.logpdf(jnp.exp(lamda), a=.5, loc=0., scale=2.) + lamda
    #     ) + gamma.logpdf(jnp.exp(tau), a=.5, loc=0., scale=2.) + tau
    #     #likelihood
    #     logit = jnp.sum(self.X * (jnp.exp(tau) * beta * jnp.exp(lamda)), axis=1)
    #     p = jnp.clip(expit(logit), a_min=1e-6, a_max=1-1e-6)
    #     # p = jsp.special.expit(logit)
    #     lprob += jnp.sum(bernoulli.logpmf(self.y, p))
    #     return lprob

    # def logprob_fn(self, params):
    #     return self.logprob(**params)
    
    # def potential_fn(self, params):
    #     return -self.logprob(**params)


class PredatorPrey:
    def __init__(self, 
        time, pred_data, prey_data,
        alpha_mean = 1., beta_mean = .05, gamma_mean = 1., delta_mean = .05,
        pred0_mean = jnp.log(10), prey0_mean = jnp.log(10), sdpred_mean = -1., sdprey_mean = -1.,
        alpha_scale = .5, beta_scale = .05, gamma_scale = .5, delta_scale = .05,
        pred0_scale = 1., prey0_scale = 1., sdpred_scale = 1., sdprey_scale = 1.,
    ) -> None:
        self.time = time
        self.data = jnp.stack([prey_data, pred_data]).T
        self.init_mean = jnp.stack([prey0_mean, pred0_mean])
        self.init_scale = jnp.stack([prey0_scale, pred0_scale])
        self.param_mean = jnp.array([
            alpha_mean, beta_mean,
            gamma_mean, delta_mean,
        ])
        self.param_scale = jnp.array([
            alpha_scale, beta_scale,
            gamma_scale, delta_scale,
        ])
        self.sd_mean = jnp.array([sdprey_mean, sdpred_mean])
        self.sd_scale = jnp.array([sdprey_scale, sdpred_scale])

    def model(self, y=None):
        pp_init = numpyro.sample('prey_pred_init', dist.LogNormal(self.init_mean, self.init_scale))
        # id_print(pp_init)
        param = numpyro.sample('param:abgd',
            dist.TruncatedNormal(low=0., loc=self.param_mean, scale=self.param_scale)
            # dist.LogNormal(loc=self.param_mean, scale=self.param_scale)
        )
        ts = jnp.arange(float(self.time.shape[0]))
        # id_print(param)
        pp = odeint(dpp_dt, pp_init, ts, *param, 
            rtol=1e-6, 
            atol=1e-5, 
            mxstep=1000
        )
        pp = jnp.clip(pp, a_min=1e-6)

        # term = diffrax.ODETerm(dxy_dt)
        # solver = diffrax.Dopri5()
        # # solver = diffrax.Tsit5()
        # # solver = diffrax.Dopri8()
        # id_print(pp_init)
        # id_print(param)
        # pp = diffrax.diffeqsolve(term, solver, t0=0, t1=self.time.shape[0], dt0=1, y0=pp_init, args=param).ys
        # pp = jnp.clip(pp, a_min=1e-6)
        # id_print(jnp.min(pp))
        # id_print(jnp.max(pp))
        # pp = simple_euler(pp_init, self.time.shape[0], *param)
        sd = numpyro.sample('sd', dist.LogNormal(self.sd_mean, self.sd_scale))
        # id_print(sd)
        numpyro.sample('y', dist.LogNormal(jnp.log(pp), sd), obs=y)

    def initialize_model(self, rng_key, n_chain):
    
    #     (init_params, *_), self.potential_fn, *_ = initialize_model(
    #         rng_key, self.model, model_kwargs={'y': self.data},
    #     )
    #     kchain = jax.random.split(rng_key, n_chain)
    #     flat, unravel_fn = jax.flatten_util.ravel_pytree(init_params)
    #     self.init_params = jax.vmap(lambda k: unravel_fn(jax.random.normal(k, flat.shape)))(kchain)
    #     # self.init_params = jax.vmap(lambda k: unravel_fn(flat))(kchain)

    # def logprob_fn(self, params):
    #     return -self.potential_fn(params)

        self.init_params = {
            name: jax.random.normal(k, shape=(n_chain,)) * .1 for name, k in zip(
                ['lalpha', 'lbeta', 'lgamma', 'ldelta', 'lsd_pred', 'lsd_prey', 'lpred_init', 'lprey_init'],
                jax.random.split(rng_key, 8)
            )
        }

    def logprob(self, 
        lpred_init, lprey_init,
        lalpha, lbeta, lgamma, ldelta, 
        lsd_pred, lsd_prey,
    ):
        lsd = jnp.array([lsd_pred, lsd_prey])
        logsd_norm = (lsd - self.sd_mean) / self.sd_scale
        sd = jnp.exp(lsd)

        lparam = jnp.array([lalpha, lbeta, lgamma, ldelta])
        param = jnp.exp(lparam)
        param_norm = (param - self.param_mean) / self.param_scale

        lxy_init = jnp.array([lpred_init, lprey_init])
        init_norm = (lxy_init - self.init_mean) / self.init_scale
        xy_init = jnp.exp(lxy_init)

        # ts = jnp.arange(float(self.time.shape[0]))
        # xy = odeint(dpp_dt, 
        #     xy_init, ts, *param,
        #     rtol=1e-6, atol=1e-5, 
        #     mxstep=1000,
        # )
        # xy = jnp.clip(xy, a_min=1e-6)
        term = diffrax.ODETerm(dxy_dt)
        # solver = diffrax.Dopri5()
        solver = diffrax.Tsit5()
        # solver = diffrax.Dopri8()
        # id_print(xy_init)
        # id_print(param)
        xy = diffrax.diffeqsolve(term, solver, t0=0, t1=self.time.shape[0], dt0=1, y0=xy_init, args=param).ys
        xy = jnp.clip(xy, a_min=1e-6)
        logdata_norm = (jnp.log(self.data) - jnp.log(xy)) / sd
        return (
            -.5 * jnp.dot(logsd_norm, logsd_norm) #- jnp.sum(sd) #lognormal sd
            + jnp.sum( #truncated normal params (assume params within bounds)
                # - jnp.log(self.param_scale) 
                + jax.vmap(norm.logpdf)(param_norm) 
                # - jax.vmap(lambda x: jnp.log(1. - norm.cdf(x)))(-self.param_mean / self.param_scale)
            )
            + jnp.sum(lparam) #vol change from evaluating param at log
            -.5 * jnp.dot(init_norm, init_norm) #- jnp.sum(xy_init) #lognormal pred-prey at time 0
            -.5 * jnp.sum(logdata_norm ** 2) - jnp.sum(lsd) * self.data.shape[0] #lognormal likelihood
        )

    def logprob_fn(self, params):
        return self.logprob(**params)
    
    # def potential_fn(self, params):
    #     return -self.logprob(**params)


class Distribution(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def logprob(self, x1, x2):
        """defines the log probability function"""

    def logprob_fn(self, x):
        return self.logprob(**x)

    @abc.abstractmethod
    def initialize_model(self, rng_key, n_chain):
        """defines the initialization of paramters"""


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

class Banana(BiDistribution):
    def logprob(self, x1, x2):
        return norm.logpdf(x1, 0.0, jnp.sqrt(8.0)) + norm.logpdf(
            x2, 1 / 4 * x1**2, 1.0
        )

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


def dpp_dt(xy, t, alpha, beta, gamma, delta):
    """Lotka-Volterra equations"""
    prey, pred = xy[0], xy[1]
    dprey = (alpha - beta * pred) * prey
    dpred = (-gamma + delta * prey) * pred
    return jnp.stack([dprey, dpred])

def dxy_dt(t, xy, theta):
    """Lotka-Volterra equations"""
    prey, pred = xy[0], xy[1]
    alpha, beta, gamma, delta = (
        theta[..., 0],
        theta[..., 1],
        theta[..., 2],
        theta[..., 3],
    )
    dprey = (alpha - beta * pred) * prey
    dpred = (-gamma + delta * prey) * pred
    return jnp.stack([dprey, dpred])

def simple_euler(xy, t1, alpha, beta, gamma, delta):
    def step(carry, i):
        prey, pred = carry
        prey += (alpha - beta * pred) * prey
        pred += (-gamma + delta * prey) * pred
        return (prey, pred), jnp.array([prey, pred])
    _, prey_pred = jax.lax.scan(step, (xy[0], xy[1]), jnp.arange(t1-1))
    return jnp.vstack([xy, prey_pred])
