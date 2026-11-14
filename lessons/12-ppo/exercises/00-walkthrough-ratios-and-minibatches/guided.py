import numpy as np
import rlhud


#> log π(aᵢ|sᵢ) of the action actually taken in each row of a probability table `(N, n)`.
def taken_logp(P, actions):
    rows = np.arange(len(actions))
    #> `P[rows, actions]` pairs row i with column actions[i] → shape `(N,)`.
    #> `+ 1e-12` guards against log(0) = −inf when a probability underflows to zero.
    return np.log(P[rows, actions] + 1e-12)


#> **Importance ratio** π_new / π_old from LOG-probabilities: exp(a − b) = eᵃ / eᵇ,
#> without dividing tiny numbers. Ratio 1 = the policy did not change for that sample.
def ratios(logp_new, logp_old):
    return np.exp(np.asarray(logp_new, dtype=float) - np.asarray(logp_old, dtype=float))


#> **Truncated importance sampling**: estimate π_new's mean reward from data collected
#> with π_old, capping each weight at `cap` so one lucky sample cannot dominate.
def capped_estimate(ratio, rewards, cap):
    #> `np.minimum` is the ELEMENT-wise min of two arrays (or array and number).
    #> `np.min` would instead reduce everything to ONE number — a classic mix-up.
    w = np.minimum(ratio, cap)
    return float(np.mean(w * rewards))


#> Its gradient w.r.t. logp_new. Since d ratio / d logp_new = ratio, each term contributes
#> ratio·reward / N — except where the cap is active: a flat function has zero slope.
def capped_estimate_grad(ratio, rewards, cap):
    #> `np.where(cond, a, b)` picks `a` where cond is True and `b` elsewhere, per element.
    return np.where(ratio > cap, 0.0, ratio * rewards) / len(ratio)


#> The fraction of samples whose ratio left the band [lo, hi] — a health statistic.
def fraction_outside(ratio, lo, hi):
    #> `|` is element-wise OR on boolean arrays. The parentheses are REQUIRED: `|` binds
    #> tighter than `<`, so `ratio < lo | ratio > hi` means something else entirely.
    outside = (ratio < lo) | (ratio > hi)
    #> The mean of a boolean array is the fraction of True values.
    return float(np.mean(outside))


#> **Epochs × shuffled minibatches** over one fixed batch, a dict of arrays with N rows.
#> Here the model is a linear regression y ≈ Xw, fitted on ½·mean(err²).
def minibatch_fit(batch, w, lr, epochs, size, rng):
    N = len(batch["y"])
    for epoch in range(epochs):
        #> A fresh random order every epoch: each row is used exactly once per epoch.
        order = rng.permutation(N)
        for start in range(0, N, size):
            #> Slicing past the end is fine in Python: the last minibatch is just shorter.
            mb = order[start:start + size]
            #> The SAME index array for every field keeps each row's x and y together.
            X, y = batch["x"][mb], batch["y"][mb]
            #> Divide by `len(mb)`, not `size` — the last minibatch may be smaller.
            w -= lr * X.T @ (X @ w - y) / len(mb)
    return w


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    #> Behaviour data: 2000 actions from a uniform old policy over 3 arms.
    P_old = np.full((2000, 3), 1 / 3)
    A = rng.integers(0, 3, size=2000)
    R = np.array([0.0, 1.0, 3.0])[A] + rng.normal(size=2000)
    #> A new policy that prefers arm 2: true mean reward 0.1·0 + 0.3·1 + 0.6·3 = 2.1.
    P_new = np.tile([0.1, 0.3, 0.6], (2000, 1))
    r = ratios(taken_logp(P_new, A), taken_logp(P_old, A))
    #= given
    caps = np.linspace(0.5, 2.5, 21)
    est = [capped_estimate(r, R, c) for c in caps]
    title = "Value of π_new from π_old's data"
    rlhud.plot(est, x=caps, label="capped estimate", title=title, xlabel="cap")
    rlhud.plot([2.1] * len(caps), x=caps, label="true value 2.1", title=title)
    rlhud.stat("Fraction of ratios outside [0.8, 1.2]", fraction_outside(r, 0.8, 1.2))
    rlhud.stat("Nonzero gradient entries at cap 1.5",
               int(np.count_nonzero(capped_estimate_grad(r, R, 1.5))))
    #= end
    #> Minibatch regression: 500 rows in minibatches of 64 (the last one has 52 rows).
    X = np.c_[rng.normal(size=500), np.ones(500)]
    batch = {"x": X, "y": X @ np.array([2.0, -1.0]) + 0.1 * rng.normal(size=500)}
    w = minibatch_fit(batch, np.zeros(2), lr=0.1, epochs=20, size=64, rng=rng)
    #= given
    rlhud.bars(w, labels=["slope (true 2)", "bias (true −1)"], title="Fitted weights")
    #= end
