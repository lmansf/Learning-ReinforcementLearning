import numpy as np
import rlhud
from rlenvs import GridWorld


def q_update(Q, s, a, r, s2, terminated, alpha, gamma):
    """Q-learning update in place: Q[s,a] += alpha * (r + gamma * max Q[s2] - Q[s,a]) (no bootstrap if terminated)."""
    target = r if terminated else r + gamma * np.max(Q[s2])
    Q[s, a] += alpha * (target - Q[s, a])


def plan(Q, model, n, rng, alpha, gamma):
    """n planning updates. Each one: pick a previously seen (s, a) uniformly at random from `model`,
    look up the remembered (r, s2, terminated) and apply q_update as if it had just happened."""
    if not model:
        return
    keys = list(model)
    for _ in range(n):
        s, a = keys[rng.integers(len(keys))]
        r, s2, terminated = model[(s, a)]
        q_update(Q, s, a, r, s2, terminated, alpha, gamma)


def dyna_q(env, n_episodes, n_planning, alpha=0.1, gamma=0.95, epsilon=0.1, seed=0):
    """Tabular Dyna-Q (Sutton & Barto, section 8.2).

    Returns (Q, steps, model):
      Q     : (n_states, n_actions) array
      steps : list, number of real environment steps taken in each episode
      model : dict {(s, a): (r, s2, terminated)} — the last outcome seen for each pair
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    model = {}
    steps = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        n = 0
        while True:
            a = epsilon_greedy(Q, s, epsilon, rng)                      # (a) act
            s2, r, terminated, truncated, _ = env.step(a)
            n += 1
            q_update(Q, s, a, r, s2, terminated, alpha, gamma)          # (b) direct RL
            model[(s, a)] = (r, s2, terminated)                         # (c) model learning
            plan(Q, model, n_planning, rng, alpha, gamma)               # (d) planning
            if terminated or truncated:
                break
            s = s2
        steps.append(n)
    return Q, steps, model


# ---------------------------------------------------------------- provided
def epsilon_greedy(Q, s, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))


# The maze of Sutton & Barto Figure 8.2: 6 x 9, start on the left, goal top-right. Shortest path: 14 steps.
MAZE = [
    ".......#G",
    "..#....#.",
    "S.#....#.",
    "..#......",
    ".....#...",
    ".........",
]


def make_maze():
    return GridWorld(MAZE, step_reward=0.0, rewards={"G": 1.0}, max_steps=5000)


if __name__ == "__main__":
    EPISODES, RUNS = 50, 10
    for n_plan in (0, 5, 50):
        curve = np.zeros(EPISODES)
        for run in range(RUNS):
            env = make_maze()
            _, steps, _ = dyna_q(env, EPISODES, n_plan, seed=run)
            curve += np.array(steps) / RUNS
        rlhud.plot(curve[1:], x=list(range(2, EPISODES + 1)), label=f"{n_plan} planning steps",
                   title="Dyna maze: steps per episode (avg of 10 runs)", xlabel="episode", ylabel="steps")
        rlhud.stat(f"n={n_plan}: steps in episode 2", round(float(curve[1]), 1))

    # Policy after only 2 episodes: with planning, most of the maze already points to the goal
    for n_plan in (0, 50):
        env = make_maze()
        Q, _, model = dyna_q(env, 2, n_plan, seed=0)
        arrows = env.policy_grid(Q)
        for st in range(env.n_states):            # no arrow where nothing has been learned yet
            if Q[st].max() <= 0:
                r, c = env.pos_of(st)
                arrows[r][c] = None
        rlhud.grid(env.to_grid(Q.max(axis=1)), title=f"max Q after 2 episodes, n = {n_plan}",
                   policy=arrows,
                   walls=env.walls_mask(), labels=env.labels(), cmap="heat")
    print("Planning re-uses every real transition many times: the value of the goal spreads in a few episodes.")
