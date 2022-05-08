import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.nn.initializers import glorot_normal, normal

from jax.experimental.host_callback import id_print

import optax


def affine_iaf_masks(d, n_hidden):
    masks = [jnp.zeros((d, d)).at[jnp.tril_indices(d, -1)].set(1.)]
    for _ in range(n_hidden-1):
        masks.append(jnp.zeros((d, d)).at[jnp.tril_indices(d)].set(1.))
    masks.append(jnp.vstack([
        jnp.zeros((d, d)).at[jnp.tril_indices(d)].set(1.),
        jnp.zeros((d, d)).at[jnp.tril_indices(d)].set(1.),
    ]))
    return masks

def MaskedDense(mask, W_init=glorot_normal(), b_init=normal()):
    def init_fun(rng_key, input_shape):
        k1, k2 = jrnd.split(rng_key)
        W = W_init(k1, mask.shape)
        b = b_init(k2, mask.shape[:1])
        params = (W, b)
        return input_shape[:-1] + mask.shape[:1], params
    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return jnp.dot(W * mask, inputs) + b
    return init_fun, apply_fun

def Dropout(rate):
    """Layer construction function for a dropout layer with given rate."""
    def init_fun(rng, input_shape):
        return input_shape, ()
    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.get('rng', None)
        mode = kwargs.get('mode', 'train')
        if rng is None:
            msg = ("Dropout layer requires apply_fun to be called with a PRNG key "
                  "argument. That is, instead of `apply_fun(params, inputs)`, call "
                  "it like `apply_fun(params, inputs, rng)` where `rng` is a "
                  "jax.random.PRNGKey value.")
            raise ValueError(msg)
        if mode == 'train':
            keep = jrnd.bernoulli(rng, rate, inputs.shape)
            return jnp.where(keep, inputs / rate, 0)
        else:
            return inputs
    return init_fun, apply_fun


# def optimize(init_param, solver, n_iter, *args):
#     def solver_iter(carry, _):
#         param, state = solver.update(*carry, *args)
#         return (param, state), state.error
#     (param, _), err = jax.lax.scan(solver_iter, 
#         (init_param, solver.init_state(init_param, *args)),
#         jnp.arange(n_iter))
#     return param, err
#     # param, sstate = solver.run(init_param, *args)
#     # return param, sstate.iter_num

# def optimize(init_param, loss, optim, n_iter, *args):
def optimize(init_param, loss, check, optim, tol, key, *args):
    opt_state = optim.init(init_param)

    # def step(carry, _):
    #     params, opt_state = carry
    #     loss_value, grads = jax.value_and_grad(loss)(params, *args)
    #     updates, opt_state_ = optim.update(grads, opt_state, params)
    #     params_ = optax.apply_updates(params, updates)
    #     return jax.lax.cond(
    #         jnp.isfinite(loss_value) & jnp.isfinite(jax.flatten_util.ravel_pytree(grads)[0]).all(),
    #         lambda _: ((params_, opt_state_), loss_value),
    #         lambda _: ((params, opt_state), jnp.nan),
    #         None,
    #     )
    # (param, _), err = jax.lax.scan(step, (init_param, opt_state), jnp.arange(n_iter))
    # return param, err

    # def step(carry, _):
    def while_fn(bool_carry):
        _, prev_var, carry = bool_carry
        # args = arg_gn(batch, params)
        def step_epoch(carry, arg):
            k, params, opt_state = carry
            k, ki = jrnd.split(k)
            loss_value, grads = jax.value_and_grad(loss)(params, ki, *arg)
            updates, opt_state_ = optim.update(grads, opt_state, params)
            params_ = optax.apply_updates(params, updates)
            return jax.lax.cond(
                jnp.isfinite(loss_value) & jnp.isfinite(jax.flatten_util.ravel_pytree(grads)[0]).all(),
                lambda _: ((k, params_, opt_state_), loss_value),
                lambda _: ((k, params, opt_state), jnp.nan),
                None,
            )
        (key, params, state), loss_value = jax.lax.scan(step_epoch, carry, args)
        # avg_var = jnp.mean(jax.vmap(lambda *arg: check(params, *arg))(*args))
        avg_var = check(params, key, *args)
        # id_print(avg_var)
        stop = (avg_var >= tol) & (jnp.abs(avg_var - prev_var) >= 1e-6)
        return stop, avg_var, (key, params, state)
        # return jax.lax.scan(step_epoch, carry, args)
    # (param, _), err = jax.lax.scan(step, (init_param, opt_state), jnp.arange(n_iter))
    *_, (_, param, _) = jax.lax.while_loop(lambda c: c[0], while_fn, (True, 0., (key, init_param, opt_state)))
    return param, None#err


    