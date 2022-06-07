import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.nn.initializers import glorot_normal, normal

from jax.experimental.host_callback import id_print
from numpy import std

import optax
import haiku as hk
import distrax as dx

def affine_iaf_masks(d, n_hidden):
    masks = [jnp.zeros((d, d)).at[jnp.tril_indices(d, -1)].set(1.)]
    for _ in range(n_hidden-1):
        masks.append(jnp.zeros((d, d)).at[jnp.tril_indices(d)].set(1.))
    masks.append(jnp.vstack([
        jnp.zeros((d, d)).at[jnp.tril_indices(d)].set(1.),
        jnp.zeros((d, d)).at[jnp.tril_indices(d)].set(1.),
    ]))
    return masks

class MaskedLinear(hk.Module):
    def __init__(self, mask, 
        w_init=hk.initializers.VarianceScaling(scale=.1),
        # w_init=hk.initializers.TruncatedNormal(stddev=.1), 
        # w_init=jnp.zeros,
        b_init=hk.initializers.RandomNormal(stddev=.1), 
        # b_init=jnp.zeros
    ):
        super().__init__()
        self._mask = mask
        self._w_init = w_init
        self._b_init = b_init
    def __call__(self, x):
        mask_shape = self._mask.shape
        w = hk.get_parameter('w',
            shape=mask_shape,
            init=self._w_init
            # init=hk.initializers.TruncatedNormal(stddev=1. / jnp.sqrt(mask_shape[0])),
        )
        out = jnp.dot(w * self._mask, x)
        b = hk.get_parameter('b', shape=mask_shape[:-1], init=self._b_init)
        # b = jnp.broadcast_to(b, out.shape)
        return out + b

class Autoregressive(dx.Bijector):
    def __init__(self, d, conditioner, bijector):
        self._conditioner = conditioner
        self._bijector = bijector
        super().__init__(event_ndims_in=1)
    
    def forward_and_log_det(self, u):
        params = self._conditioner(u)
        x, log_d = self._bijector(params).forward_and_log_det(u)
        return x, jnp.sum(log_d)

    def inverse_and_log_det(self, x):
        params = self._conditioner(x)
        u, log_d = self._bijector(params).inverse_and_log_det(x)
        return u, jnp.sum(log_d)


def optimize(init_param, loss_check, optim, tol, maxiter, key, X, batch_iter, batch_size):
    opt_state = optim.init(init_param)
    loss, check = loss_check

    def while_fn(var_carry):
        _, carry = var_carry
        def step_epoch(carry, x):
            k, i, params, opt_state = carry
            k, ki = jrnd.split(k)
            loss_value, grads = jax.value_and_grad(loss)(params, ki, x, batch_size)
            updates, opt_state_ = optim.update(grads, opt_state, params)
            params_ = optax.apply_updates(params, updates)
            return jax.lax.cond(
                jnp.isfinite(loss_value) & jnp.isfinite(jax.flatten_util.ravel_pytree(grads)[0]).all(),
                lambda _: ((k, i+1, params_, opt_state_), loss_value),
                lambda _: ((k, i+1, params, opt_state), jnp.nan),
                None,
            )
        x = jax.tree_map(lambda x: jrnd.choice(carry[0], x, (batch_iter, batch_size), False), X)
        (key, i, params, state), loss_value = jax.lax.scan(step_epoch, carry, x)
        var = check(params, key, X, batch_size * batch_iter)
        # id_print(var)
        return var, (key, i, params, state)
    stop_fn = lambda c: (c[0] >= tol) & (c[1][1] < maxiter) 
    var, (*_, param, _) = jax.lax.while_loop(stop_fn, while_fn, (1e6, (key, 0, init_param, opt_state)))
    return param, var


def optimize2(init_param, loss, check, optim, key, X, batch_iter, batch_size, n_iter):
    opt_state = optim.init(init_param)

    def iter_fn(carry, _):
        key, init_param, opt_state = carry
        def batch_fn(carry, x):
            k, params, opt_state = carry
            k, ki = jrnd.split(k)
            loss_value, grads = jax.value_and_grad(loss)(params, ki, x)
            updates, opt_state_ = optim.update(grads, opt_state, params)
            params_ = optax.apply_updates(params, updates)
            return jax.lax.cond(
                jnp.isfinite(loss_value) & jnp.isfinite(jax.flatten_util.ravel_pytree(grads)[0]).all(),
                lambda _: ((k, params_, opt_state_), loss_value),
                lambda _: ((k, params, opt_state), jnp.nan),
                None,
            )
        x = jax.tree_map(lambda x: jrnd.choice(key, x, (batch_iter, batch_size), False), X)
        (key, params, state), loss_value = jax.lax.scan(batch_fn, (key, init_param, opt_state), x)
        var = check(params, key, X)
        return (key, params, state), var
    (_, params, _), var = jax.lax.scan(iter_fn, (key, init_param, opt_state), jnp.arange(n_iter))
    return params, var
    