import numpy as np
import rlhud
from corerunner import CoreRunner


# ---------------------------------------------------------------- state design
# The env exposes its raw internals:
#   env.pos            (row, col) of the drone
#   env.collected      list[bool], one per core
#   env.t              steps taken this episode
#   env.period         sentries repeat their motion every `period` steps
#   env.h, env.w       grid size;  len(env.core_list) = number of cores

def n_states_for(env):
    """How many distinct encoded states encode_state can return."""
    # TODO
    raise NotImplementedError


def encode_state(env):
    """Pack everything that matters for the future into ONE integer in [0, n_states_for(env))."""
    # TODO: combine position, which cores are collected (a bitmask) and the sentry phase
    raise NotImplementedError


def encode_position_only(env):
    """A deliberately BROKEN state (ignores sentries) — for comparison in the demo."""
    r, c = env.pos
    mask = sum(int(g) << i for i, g in enumerate(env.collected))
    return (r * env.w + c) * (2 ** len(env.core_list)) + mask


# ---------------------------------------------------------------- learning
def train_patrol(episodes=3000, seed=0, encoder=encode_state, n_states=None,
                 alpha=0.5, gamma=0.99, curve=None):
    """Q-learning on level 3 using YOUR state encoder (env observations are ignored)."""
    rng = np.random.default_rng(seed)
    env = CoreRunner(level=3)
    Q = np.zeros((n_states or n_states_for(env), env.n_actions))
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.6 * episodes))
        env.reset()
        s, G = encoder(env), 0.0
        # TODO: the Q-learning episode loop — after env.step(a), compute s2 = encoder(env)
        raise NotImplementedError
        if curve is not None:
            curve.append(G)
    return Q


def play_greedy(Q, encoder=encode_state, record=False):
    env = CoreRunner(level=3)
    env.reset()
    frames, G = [env.snapshot()], 0.0
    while True:
        _, r, term, trunc, info = env.step(int(np.argmax(Q[encoder(env)])))
        G += r
        if record:
            frames.append(env.snapshot())
        if term or trunc:
            return G, info["event"], frames


if __name__ == "__main__":
    env = CoreRunner(level=3)
    for name, enc, n in [("position + cores (no sentry phase)", encode_position_only, env.h * env.w * 8),
                         ("full Markov state", encode_state, n_states_for(env))]:
        curve = []
        Q = train_patrol(3000, seed=0, encoder=enc, n_states=n, curve=curve)
        G, event, frames = play_greedy(Q, enc, record=True)
        rlhud.plot(curve, label=name, title="Patrol Deck · training return", smooth=50)
        rlhud.stat(name, f"{G:.1f} ({event})")
        rlhud.replay(frames, title=f"Greedy · {name}")
