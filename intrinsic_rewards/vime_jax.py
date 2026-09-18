"""
VIME: Variational Information Maximizing Exploration (Houthooft et al., 2016).

A Bayesian neural network (Blundell et al.'s Bayes-by-Backprop, fully-factorized
Gaussian weight posterior q(theta) = N(mu, softplus(rho)^2)) is trained online to
predict the environment's forward dynamics (delta-observation given obs, action).

The intrinsic reward for a transition is the information gain it would produce
about the dynamics model: the KL divergence between the posterior before and
after a single gradient step on that one transition's likelihood. This follows
the common practical approximation used in most VIME reimplementations (a plain
gradient step rather than the paper's full Hessian-weighted Newton step), with
the resulting KL divergence normalized by a running median (per the paper) to
keep the reward scale stable.
"""

import jax
import jax.numpy as jnp


def init_bnn_params(rng, layer_sizes, init_rho=-3.0):
    """Initialize the fully-factorized Gaussian posterior (mu, rho) for each layer."""
    mus, rhos = [], []
    keys = jax.random.split(rng, len(layer_sizes) - 1)
    for k, fan_in, fan_out in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        w_mu = jax.random.normal(k, (fan_in, fan_out)) * jnp.sqrt(1.0 / fan_in)
        b_mu = jnp.zeros((fan_out,))
        w_rho = jnp.full((fan_in, fan_out), init_rho)
        b_rho = jnp.full((fan_out,), init_rho)
        mus.append({'w': w_mu, 'b': b_mu})
        rhos.append({'w': w_rho, 'b': b_rho})
    return {'mu': mus, 'rho': rhos}


def sample_weights(params, rng):
    """Draw one Monte Carlo weight sample via the reparameterization trick."""
    mus, rhos = params['mu'], params['rho']
    keys = jax.random.split(rng, 2 * len(mus))
    weights = []
    for i, (mu, rho) in enumerate(zip(mus, rhos)):
        k_w, k_b = keys[2 * i], keys[2 * i + 1]
        std_w = jax.nn.softplus(rho['w'])
        std_b = jax.nn.softplus(rho['b'])
        w = mu['w'] + std_w * jax.random.normal(k_w, mu['w'].shape)
        b = mu['b'] + std_b * jax.random.normal(k_b, mu['b'].shape)
        weights.append({'w': w, 'b': b})
    return weights


def bnn_forward(weights, x):
    """Forward pass for a single example (no batch dimension)."""
    h = x
    n_layers = len(weights)
    for i, layer in enumerate(weights):
        h = h @ layer['w'] + layer['b']
        if i < n_layers - 1:
            h = jax.nn.relu(h)
    return h


def nll_loss(params, rng, obs_action, target_delta, out_sigma):
    """Negative Gaussian log-likelihood (up to a constant) for a single transition."""
    weights = sample_weights(params, rng)
    pred = bnn_forward(weights, obs_action)
    return jnp.sum((pred - target_delta) ** 2) / (2 * out_sigma ** 2)


def kl_divergence_params(params_new, params_old):
    """KL[q(theta; params_new) || q(theta; params_old)] summed over all weights."""
    total = 0.0
    for mu_n, rho_n, mu_o, rho_o in zip(
        params_new['mu'], params_new['rho'], params_old['mu'], params_old['rho']
    ):
        for key in ('w', 'b'):
            sigma_n = jax.nn.softplus(rho_n[key])
            sigma_o = jax.nn.softplus(rho_o[key])
            kl = (
                jnp.log(sigma_o / sigma_n)
                + (sigma_n ** 2 + (mu_n[key] - mu_o[key]) ** 2) / (2 * sigma_o ** 2)
                - 0.5
            )
            total = total + jnp.sum(kl)
    return total


def kl_to_prior(params, prior_std):
    """KL[q(theta) || N(0, prior_std^2)] summed over all weights."""
    total = 0.0
    for mu, rho in zip(params['mu'], params['rho']):
        for key in ('w', 'b'):
            sigma = jax.nn.softplus(rho[key])
            kl = (
                jnp.log(prior_std / sigma)
                + (sigma ** 2 + mu[key] ** 2) / (2 * prior_std ** 2)
                - 0.5
            )
            total = total + jnp.sum(kl)
    return total


def probe_kl_reward(params, rng, obs_action, target_delta, step_size, out_sigma):
    """Information-gain intrinsic reward for a single transition.

    Takes one gradient step on the transition's negative log-likelihood and
    returns the resulting KL divergence between the new and old posteriors.
    """
    grads = jax.grad(nll_loss)(params, rng, obs_action, target_delta, out_sigma)
    new_params = jax.tree.map(lambda p, g: p - step_size * g, params, grads)
    return kl_divergence_params(new_params, params)


def elbo_loss(params, rng, obs_action_batch, target_delta_batch, out_sigma, prior_std, n_total):
    """Negative variational lower bound for a batch, used to actually train the BNN."""
    weights = sample_weights(params, rng)
    preds = jax.vmap(bnn_forward, in_axes=(None, 0))(weights, obs_action_batch)
    nll = jnp.sum((preds - target_delta_batch) ** 2, axis=-1) / (2 * out_sigma ** 2)
    kl_prior = kl_to_prior(params, prior_std)
    return jnp.mean(nll) + kl_prior / n_total
