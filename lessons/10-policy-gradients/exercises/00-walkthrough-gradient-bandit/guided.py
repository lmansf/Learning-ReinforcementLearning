import numpy as np
import rlhud


#> **Softmax** turns preferences into probabilities — for ONE row or a BATCH of rows.
def softmax(h):
    h = np.asarray(h, dtype=float)
    #> `exp(1000)` overflows to inf and the result becomes nan. Subtracting each row's max
    #> changes nothing (the factor cancels in the ratio) but keeps every exponent ≤ 0.
    #> `axis=-1, keepdims=True` keeps shape `(B, 1)` so it broadcasts against `(B, n)`.
    z = h - h.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


#> Draw an action index from a probability vector `p`.
def sample(p, rng):
    #> `rng.choice(n, p=p)` draws from 0 … n−1 with probabilities p (they must sum to 1).
    #> `int(...)` turns NumPy's int64 into a plain Python int.
    return int(rng.choice(len(p), p=p))


#> One-hot rows for a batch of actions: row i is all zeros except a 1 in column `A[i]`.
def one_hot(A, n):
    #> Indexing the identity matrix with an int array picks its rows:
    #> shape `(len(A), n)`.
    return np.eye(n)[A]


#> **Gradient bandit** (Sutton & Barto §2.8). Preferences H, policy π = softmax(H), and
#> ∂ log π(a) / ∂H = onehot(a) − π: up for the action taken, down for all the others.
def bandit_step(H, a, r, baseline, alpha):
    p = softmax(H)
    #> `one_hot` wants an array of actions; `[0]` takes the single row back out.
    score = one_hot(np.array([a]), len(H))[0] - p
    #> Gradient ASCENT on expected reward, so `+=`. (r − baseline) says whether this
    #> reward was better or worse than usual; without a baseline every reward pushes up.
    H += alpha * (r - baseline) * score
    return p


#> The same update for a BATCH of N pulls in several contexts: H has one row per context.
def batch_step(H, C, A, R, alpha):
    #> `H[C]` gathers each sample's row → `(N, n_actions)`; softmax works row by row.
    P = softmax(H[C])
    #> Normalised advantages: centre AND scale, so the step size does not depend on units.
    #> `+ 1e-8` avoids dividing by zero when all rewards are equal.
    adv = (R - R.mean()) / (R.std() + 1e-8)
    #> `adv[:, None]` has shape `(N, 1)`: it scales each ROW of (onehot − P) by that
    #> sample's advantage. Plain `adv`, shape `(N,)`, would line up with the COLUMNS.
    G = (one_hot(A, H.shape[1]) - P) * adv[:, None]
    #> Contexts as one-hot features X `(N, n_ctx)`. `X.T @ G` adds up the rows of G that
    #> share a context — the batch sum of outer(xᵢ, gᵢ). Divide by N for the mean.
    X = one_hot(C, H.shape[0])
    H += alpha * X.T @ G / len(A)
    return P


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    #> A 2-armed bandit: arm 1 pays 1.0 on average, arm 0 only 0.5.
    means = np.array([0.5, 1.0])
    H = np.zeros(2)
    #> The baseline is a running average of the rewards seen so far.
    baseline = 0.0
    #= given
    live = rlhud.chart("π(arm) while learning", xlabel="pull")
    #= end
    for t in range(1, 501):
        a = sample(softmax(H), rng)
        r = means[a] + rng.normal()
        p = bandit_step(H, a, r, baseline, alpha=0.1)
        #> Incremental mean: b ← b + (r − b)/t. Update it AFTER using it.
        baseline += (r - baseline) / t
        #= given
        live.add("π(arm 1)", float(p[1]))
        live.add("π(arm 0)", float(p[0]))
    live.flush()
    #= end
    #> Three contexts with a different best arm each; 64 pulls per batch.
    best = np.array([0, 1, 1])
    Hc = np.zeros((3, 2))
    for _ in range(200):
        C = rng.integers(0, 3, size=64)
        #> Sample all 64 actions from their own context's probabilities.
        A = np.array([sample(p, rng) for p in softmax(Hc[C])])
        R = (A == best[C]).astype(float) + 0.3 * rng.normal(size=64)
        batch_step(Hc, C, A, R, alpha=0.5)
    #= given
    rlhud.bars(softmax(Hc)[:, 1], labels=["ctx 0 (best: arm 0)", "ctx 1", "ctx 2"],
               title="π(arm 1 | context) after batch training")
    p1 = float(softmax(H)[1])
    rlhud.stat("π(arm 1) in the 2-armed bandit", p1, good=p1 > 0.7)
    #= end
