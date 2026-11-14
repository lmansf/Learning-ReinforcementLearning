#> `numpy` for the arrays of estimates and counts, `rlhud` for the charts.
import numpy as np
import rlhud


#> A tiny bandit of our own: each arm is a slot machine that pays **1** with probability
#> `probs[a]`, otherwise **0** (a *Bernoulli* arm). The lesson's `KArmedBandit` has the
#> same shape, but pays Gaussian rewards.
class SlotMachines:
    #> `np.asarray(..., dtype=float)` accepts a list or an array and guarantees floats.
    def __init__(self, probs, seed=None):
        self.probs = np.asarray(probs, dtype=float)
        #> `np.random.default_rng(seed)` makes a private, seedable random **Generator**.
        #> Same seed, same sequence: every experiment becomes reproducible.
        self.rng = np.random.default_rng(seed)
        #> `np.argmax` gives the index of the largest entry; `int(...)` turns NumPy's
        #> integer into a plain Python one.
        self.best_arm = int(np.argmax(self.probs))

    #> `rng.random()` is one float in [0, 1). It is below p with probability exactly p,
    #> so this comparison is a biased coin. You will use the same test for "explore with
    #> probability ε". `float(True)` is 1.0.
    def pull(self, a):
        return float(self.rng.random() < self.probs[a])


#> Greedy choice with **random tie-breaking**. `Q == Q.max()` is a boolean *mask*;
#> `np.flatnonzero` turns it into the indices where it is True, e.g. `[0, 2]`.
#> Plain `np.argmax` would always return the lowest index, a silent bias.
def greedy_arm(Q, rng):
    best = np.flatnonzero(Q == Q.max())
    #> `rng.choice(array)` picks one element uniformly. (`rng.integers(n)` would pick a
    #> random index in 0 … n-1: the other tool for exploring.)
    return int(rng.choice(best))


#> A purely greedy agent with **sample-average** estimates and an optional optimistic
#> starting value `q0`.
class GreedyAgent:
    def __init__(self, k, q0=0.0, seed=None):
        #> `np.full(k, value)` fills an array with one value: every arm starts at q0.
        self.Q = np.full(k, float(q0))
        #> Pull counts are whole numbers, so ask for `dtype=int`.
        self.N = np.zeros(k, dtype=int)
        self.rng = np.random.default_rng(seed)

    def act(self):
        return greedy_arm(self.Q, self.rng)

    #> The incremental mean: `Q ← Q + (r − Q) / N`. Count the pull **first**, so the
    #> first reward uses step size 1/1 and replaces q0 completely (optimism only lasts
    #> until an arm has been tried once).
    def update(self, a, r):
        self.N[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.N[a]


#> Many independent runs. `make_agent(seed)` is a *factory*: a function that builds a
#> fresh agent, so every run starts from scratch.
def run_many(make_agent, probs, n_runs, n_steps):
    #> One row per run, one column per arm: a 2-D array you fill row by row.
    counts = np.zeros((n_runs, len(probs)))
    totals = np.zeros(n_runs)
    for i in range(n_runs):
        #> A different seed per run (`1000 + i`): independent runs, yet repeatable.
        env = SlotMachines(probs, seed=1000 + i)
        agent = make_agent(i)
        #> The bandit loop you will write again and again: act → pull → update.
        for t in range(n_steps):
            a = agent.act()
            r = env.pull(a)
            agent.update(a, r)
            totals[i] += r
        #> Assigning a 1-D array to `counts[i]` fills the whole row i at once.
        counts[i] = agent.N
    return counts, totals


#> RUN executes this block; CHECK skips it and tests your functions instead.
if __name__ == "__main__":
    probs, runs, steps = [0.3, 0.5, 0.8], 200, 300
    rows = []
    #> `lambda s, q=q0: …` — the default argument **freezes** the current q0. Without it,
    #> every lambda would see the loop variable's last value.
    for q0 in [0.0, 2.0]:
        counts, totals = run_many(lambda s, q=q0: GreedyAgent(3, q0=q, seed=s),
                                  probs, runs, steps)
        #> `.mean(axis=0)` averages *down the rows*: one mean per arm, over all runs.
        rlhud.bars(counts.mean(axis=0), labels=["30%", "50%", "80%"], highlight=2,
                   title=f"Average pulls per arm · q0 = {q0}")
        #> A boolean mask over runs: True where the agent spent most pulls on a bad arm.
        stuck = counts[:, 2] < steps / 2
        #= given
        rows.append([q0, round(float(totals.mean()), 1), f"{100 * stuck.mean():.0f}%"])
    rlhud.table(rows, headers=["q0", "mean total reward", "runs stuck on a bad arm"],
                title="Pure greedy, with and without optimism")
    agent, env = GreedyAgent(3, q0=2.0, seed=0), SlotMachines(probs, seed=0)
    curve = []
    for t in range(steps):
        a = agent.act()
        r = env.pull(a)
        agent.update(a, r)
        curve.append(r)
    rlhud.plot(curve, label="optimistic greedy", title="Reward per pull (one run)",
               smooth=30)
    rlhud.bars(agent.Q, labels=["30%", "50%", "80%"], title="Final estimates Q(a)",
               highlight=int(np.argmax(agent.Q)))
    #= end
