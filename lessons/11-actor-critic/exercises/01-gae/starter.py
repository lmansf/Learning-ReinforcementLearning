import numpy as np
import rlhud


def td_errors(rewards, values, dones, last_value, gamma):
    """One-step TD errors δₜ = rₜ + γ (1 − doneₜ) V(sₜ₊₁) − V(sₜ) for a rollout of length T.

    values[t] = V(sₜ);  the value of the state AFTER the last step is `last_value`.
    dones[t] = 1 if the episode ended at step t (then there is no V(sₜ₊₁) to bootstrap from).
    """
    # TODO: build next_values (values shifted by one, with last_value at the end), then δ
    raise NotImplementedError


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """Generalised Advantage Estimation. Returns (advantages, returns), both shape (T,).

    Âₜ = δₜ + γλ (1 − doneₜ) Âₜ₊₁      (computed backwards, Â_T = 0)
    returnsₜ = Âₜ + V(sₜ)             (the critic's regression targets)
    """
    # TODO: walk backwards accumulating Â; reset the accumulation at episode ends
    raise NotImplementedError


if __name__ == "__main__":
    # A 10-step rollout: reward 1 per step, the episode ends at step 5 and a new one starts.
    # The critic is imperfect (it thinks every state is worth 5).
    rewards = np.ones(10)
    values = np.full(10, 5.0)
    dones = np.zeros(10); dones[4] = 1
    for lam in (0.0, 0.5, 0.9, 1.0):
        adv, _ = compute_gae(rewards, values, dones, last_value=5.0, gamma=0.9, lam=lam)
        rlhud.plot(adv, label=f"λ = {lam}", title="GAE advantages along the rollout", xlabel="step t")
    rlhud.text("λ = 0 is the one-step TD error (low variance, biased by the critic); λ = 1 is the "
               "Monte Carlo return minus V (unbiased, high variance). Note how nothing leaks across "
               "the episode boundary after step 4.", title="Reading the chart")
