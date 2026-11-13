import numpy as np
import rlhud
from rlenvs import GridWorld


def policy_model(env, policy):
    """Collapse the MDP under a fixed policy into a Markov reward process.

    policy[s, a] = π(a | s)   (an n_states × n_actions array of probabilities)

    Returns (P_pi, r_pi):
        P_pi[s, s'] = Σ_a π(a|s) · p(s' | s, a)      — only NON-terminating transitions
        r_pi[s]     = Σ_a π(a|s) · Σ p · r            — expected immediate reward
    Transitions with terminated=True contribute their reward but no successor,
    so terminal (and wall) rows of P_pi stay all-zero.
    """
    n = env.n_states
    P_pi = np.zeros((n, n))
    r_pi = np.zeros(n)
    # TODO: loop over states, actions and outcomes; accumulate π(a|s)·prob into
    #       r_pi (times r) and P_pi (only when the transition does not terminate)
    raise NotImplementedError
    return P_pi, r_pi


def solve_v(env, policy, gamma):
    """Exact v_π from the Bellman expectation equation in matrix form:

        v = r_π + γ P_π v    ⇔    (I − γ P_π) v = r_π
    """
    # TODO: build the model, then one call to np.linalg.solve
    raise NotImplementedError


def uniform_policy(env):
    return np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)


if __name__ == "__main__":
    env = GridWorld(["...G",
                     ".#.X",
                     "S..."], slip=0.2, step_reward=-0.04)
    gamma = 0.9
    v_rand = solve_v(env, uniform_policy(env), gamma)
    rlhud.grid(env.to_grid(v_rand), title="v_π for the uniform random policy (γ = 0.9, slip 0.2)",
               walls=env.walls_mask(), labels=env.labels())

    # A hand-written policy: go up the left side, then right along the top.
    UP, RIGHT, LEFT = 0, 1, 3
    acts = np.full(env.n_states, RIGHT)
    for r, c in [(2, 0), (1, 0), (1, 2), (2, 2)]:
        acts[env.state_of(r, c)] = UP
    acts[env.state_of(2, 3)] = LEFT
    det = np.eye(4)[acts]
    v_det = solve_v(env, det, gamma)
    rlhud.grid(env.to_grid(v_det), title="v_π for a hand-written policy",
               policy=env.policy_grid(acts), walls=env.walls_mask(), labels=env.labels())
    rlhud.stat("v_random(start)", float(v_rand[env.start]))
    rlhud.stat("v_hand(start)", float(v_det[env.start]))

    # How does far-sightedness change the value of the start state?
    gammas = np.linspace(0.0, 0.99, 34)
    rlhud.plot([float(solve_v(env, det, g)[env.start]) for g in gammas], x=gammas.tolist(),
               label="hand-written policy", title="v_π(start) as γ varies", xlabel="γ")
