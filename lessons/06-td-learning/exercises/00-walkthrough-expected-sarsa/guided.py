import numpy as np
import rlhud
from rlenvs import GridWorld


#> The **probabilities** of an ε-greedy policy in one state: ε/k for every action, plus
#> the remaining 1 − ε on the greedy one. `q` is one row of the Q table.
def egreedy_probs(q, epsilon):
    k = len(q)
    probs = np.full(k, epsilon / k)
    #> `np.argmax` picks one greedy action (ties → lowest index: fine for a target).
    probs[np.argmax(q)] += 1.0 - epsilon
    return probs


#> **Expected SARSA**: like SARSA, but instead of the sampled next action A' it averages
#> over all next actions with their ε-greedy probabilities. It updates `Q` in place and
#> returns the TD error δ.
def expected_sarsa_update(Q, s, a, r, s2, terminated, alpha, gamma, epsilon):
    #> `Q[s2]` is the row of action values of the next state. The dot product `@` with
    #> the probabilities gives E[Q(S', A')]. (`Q[s2]` would also work if Q were a dict
    #> `{state: array}`; then you'd write `Q[s][a]` instead of `Q[s, a]`.)
    v_next = egreedy_probs(Q[s2], epsilon) @ Q[s2]
    #> A terminal S' has no future: then the TD **target** is just r. Forgetting this
    #> bootstraps from a state where nothing more can happen.
    target = r if terminated else r + gamma * v_next
    #> The TD error: target − current estimate. Index a 2-D array with `Q[s, a]`.
    delta = target - Q[s, a]
    #> Move a fraction α of the way towards the target. `+=` on `Q[s, a]` changes the
    #> caller's table, so Q need not be returned. With a state-value array the same
    #> three lines (`V[s]` instead of `Q[s, a]`) are TD(0) prediction.
    Q[s, a] += alpha * delta
    return delta


#> The training loop. Behaviour: ε-greedy with random tie-breaking; learning: the update.
def train(env, n_episodes, alpha=0.5, gamma=1.0, epsilon=0.1, seed=0):
    rng = np.random.default_rng(seed)
    #> One row per state, one column per action, all zeros.
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        #> Accumulate this episode's total reward for the learning curve.
        G = 0.0
        while True:
            #> Explore with probability ε (`rng.random() < epsilon`), else act greedily —
            #> breaking ties at random, or an all-zero table always picks action 0.
            if rng.random() < epsilon:
                a = int(rng.integers(env.n_actions))
            else:
                a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
            s2, r, terminated, truncated, _ = env.step(a)
            G += r
            expected_sarsa_update(Q, s, a, r, s2, terminated, alpha, gamma, epsilon)
            #> Stop on either flag; otherwise **move on**: S' becomes the new S.
            if terminated or truncated:
                break
            s = s2
        returns.append(G)
    return Q, returns


if __name__ == "__main__":
    #> A pit `X` right next to the shortest path, so exploring has a price.
    env = GridWorld(["S...", ".XX.", "...G"], step_reward=-1.0,
                    rewards={"G": 0.0, "X": -20.0})
    Q, returns = train(env, 300, alpha=0.5, epsilon=0.1, seed=0)
    rlhud.plot(returns, label="Expected SARSA", title="Return per episode", smooth=20)
    #> `Q.max(axis=1)` = best value per state; `np.argmax(Q, axis=1)` = greedy action.
    rlhud.grid(env.to_grid(Q.max(axis=1)), title="max_a Q(s, a) and the greedy policy",
               policy=env.policy_grid(np.argmax(Q, axis=1)), labels=env.labels())
    #= given
    rlhud.stat("Mean return, last 50 episodes", round(float(np.mean(returns[-50:])), 2))
    s, _ = env.reset()
    rlhud.bars(Q[s], labels=["UP", "RIGHT", "DOWN", "LEFT"], title="Q(start, a)",
               highlight=int(np.argmax(Q[s])))
    #= end
