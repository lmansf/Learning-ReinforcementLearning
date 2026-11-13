import numpy as np
import rlhud


def ppo_clip_loss(logp_new, logp_old, adv, clip_eps=0.2):
    """PPO's clipped surrogate, written as a LOSS to minimise, and its gradient.

    ratio rₜ = π_new(aₜ|sₜ) / π_old(aₜ|sₜ) = exp(logp_new − logp_old)
    loss     = −mean_t[ min( rₜ Âₜ,  clip(rₜ, 1−ε, 1+ε) Âₜ ) ]
    returns (loss, dloss/dlogp_new) — the gradient has the shape of logp_new.

    Where the clipped term is the smaller one (Â > 0 and r > 1+ε, or Â < 0 and r < 1−ε)
    the objective is flat in θ, so the gradient is 0. Elsewhere d(r Â)/dlogp_new = r Â.
    """
    # TODO
    # 1) ratio = exp(logp_new − logp_old)
    # 2) loss = −mean(min(ratio·adv, clip(ratio, 1−ε, 1+ε)·adv))
    # 3) grad = −ratio·adv / N, except 0 where the clipped term is the active (flat) one
    raise NotImplementedError


def ppo_diagnostics(logp_new, logp_old, clip_eps=0.2):
    """Two numbers every PPO implementation logs. Returns (approx_kl, clip_fraction).

    approx_kl     = mean( (r − 1) − log r )     — a low-variance estimate of KL(π_old ‖ π_new), ≥ 0
    clip_fraction = fraction of samples with |r − 1| > ε
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    # The objective for ONE sample as a function of the ratio r, for a positive and a negative
    # advantage. Plain policy gradient keeps rising forever; PPO's objective goes flat once the
    # policy has moved "far enough" in the helpful direction.
    rs = np.linspace(0.025, 2.0, 80)
    for A in (1.0, -1.0):
        obj = [-ppo_clip_loss(np.log([r]), [0.0], [A])[0] for r in rs]
        rlhud.plot(obj, x=rs, label=f"PPO, Â = {A:+.0f}", title="Surrogate objective vs ratio r", xlabel="r")
        rlhud.plot(rs * A, x=rs, label=f"unclipped r·Â, Â = {A:+.0f}", title="Surrogate objective vs ratio r", xlabel="r")
        grads = [-ppo_clip_loss(np.log([r]), [0.0], [A])[1][0] for r in rs]
        rlhud.plot(grads, x=rs, label=f"Â = {A:+.0f}", title="d objective / d log π_new", xlabel="r")
    kl, cf = ppo_diagnostics(np.log([0.5, 0.7, 0.9]), np.log([0.4, 0.7, 0.6]))
    rlhud.stat("approx KL (example)", kl)
    rlhud.stat("clip fraction (example)", cf)
