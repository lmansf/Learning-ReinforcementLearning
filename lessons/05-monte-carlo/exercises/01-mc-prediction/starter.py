from collections import defaultdict

import rlhud
from rlenvs import Blackjack


def generate_episode(env, policy, seed=None):
    """Play one episode with `policy(obs) -> action`.

    Returns a list of (state, action, reward) triples, where `reward` is the
    reward received *after* taking `action` in `state`:  [(S0, A0, R1), (S1, A1, R2), ...]
    """
    # TODO: reset (passing seed), then loop: pick an action, step, record (obs, action, reward)
    #       until terminated or truncated.
    raise NotImplementedError


def mc_prediction(env, policy, n_episodes, gamma=1.0, first_visit=True, seed=None):
    """Monte Carlo estimate of v_pi. Returns a dict {state: estimated value}.

    first_visit=True  -> only the first occurrence of a state in an episode counts
    first_visit=False -> every occurrence counts (every-visit MC)
    `seed` is passed to env.reset for the first episode only.
    """
    # TODO:
    #   for each episode (seed only on the first reset):
    #     1. generate it
    #     2. walk it BACKWARDS keeping G = r + gamma * G
    #     3. add G to returns_sum[s] / returns_count[s] — for first-visit, only when t is the
    #        first time s appears in the episode
    #   return {s: average return}
    raise NotImplementedError


def stick_on_20(obs):
    """The classic 'naive' Blackjack policy: hit until you reach 20 or 21."""
    player_sum, dealer_card, usable_ace = obs
    return 0 if player_sum >= 20 else 1


if __name__ == "__main__":
    env = Blackjack(seed=0)
    N = 50_000
    V = mc_prediction(env, stick_on_20, N, seed=0)

    # Lay V out like Sutton & Barto Figure 5.1: rows = player sum 21..12, cols = dealer A..10
    sums = list(range(21, 11, -1))
    dealer = list(range(1, 11))
    for usable in (False, True):
        grid = [[V.get((p, d, usable), 0.0) for d in dealer] for p in sums]
        labels = [[f"{V.get((p, d, usable), 0.0):.2f}" for d in dealer] for p in sums]
        rlhud.grid(grid, title=f"v_π, stick on 20 — {'usable' if usable else 'no usable'} ace "
                               f"(rows 21→12, cols dealer A→10)", labels=labels)
    rlhud.stat("Episodes", N)
    rlhud.stat("States seen", len(V))
    rlhud.stat("V(20, 10, no ace)", round(V.get((20, 10, False), float("nan")), 3))
    rlhud.stat("V(15, 10, no ace)", round(V.get((15, 10, False), float("nan")), 3))
    print("Sticking on 20 is great if you get there; every sum below 20 is a losing position.")
