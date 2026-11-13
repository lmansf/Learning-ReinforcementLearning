import numpy as np

td_errors = require("td_errors")
compute_gae = require("compute_gae")

# rollout of 3 steps, no episode end, gamma 0.5
r = [1.0, 0.0, 2.0]; v = [1.0, 2.0, 1.0]; d = [0, 0, 0]
deltas = np.asarray(td_errors(r, v, d, last_value=4.0, gamma=0.5))
# δ0 = 1 + .5·2 − 1 = 1 ; δ1 = 0 + .5·1 − 2 = −1.5 ; δ2 = 2 + .5·4 − 1 = 3
check(approx(deltas, [1.0, -1.5, 3.0]), "td_errors by hand (bootstraps from last_value at the end)",
      f"expected [1, −1.5, 3], got {deltas.tolist()}")
deltas = np.asarray(td_errors(r, v, [0, 1, 0], last_value=4.0, gamma=0.5))
check(approx(deltas, [1.0, -2.0, 3.0]), "td_errors: no bootstrap after a terminal step",
      f"step 1 ends an episode → δ1 = 0 − 2 = −2. Got {deltas.tolist()}")

adv, ret = compute_gae(r, v, d, last_value=4.0, gamma=0.5, lam=0.5)
adv, ret = np.asarray(adv), np.asarray(ret)
# Â2 = 3 ; Â1 = −1.5 + .25·3 = −0.75 ; Â0 = 1 + .25·(−0.75) = 0.8125
check(approx(adv, [0.8125, -0.75, 3.0]), "GAE by hand: Âₜ = δₜ + γλ Âₜ₊₁",
      f"expected [0.8125, −0.75, 3.0], got {adv.tolist()}")
check(approx(ret, adv + np.array(v)), "returns = advantages + values", f"got {ret.tolist()}")

adv, _ = compute_gae(r, v, [0, 1, 0], last_value=4.0, gamma=0.5, lam=0.5)
# Â2 = 3 ; Â1 = −2 (episode ended: no accumulation) ; Â0 = 1 + .25·(−2) = 0.5
check(approx(adv, [0.5, -2.0, 3.0]), "GAE does not leak across episode boundaries",
      f"expected [0.5, −2, 3], got {np.asarray(adv).tolist()} — multiply the carried term by (1 − doneₜ)")

rng = np.random.default_rng(0)
T = 20
r = rng.normal(size=T); v = rng.normal(size=T); d = (rng.random(T) < 0.15).astype(float); lv = 0.7
adv0, _ = compute_gae(r, v, d, lv, gamma=0.9, lam=0.0)
check(approx(adv0, td_errors(r, v, d, lv, 0.9)), "λ = 0 gives the one-step TD errors")
adv1, ret1 = compute_gae(r, v, d, lv, gamma=0.9, lam=1.0)
G, mc = lv, np.zeros(T)
for t in reversed(range(T)):
    G = r[t] + 0.9 * (1 - d[t]) * G
    mc[t] = G
check(approx(ret1, mc, 1e-8), "λ = 1 gives Monte Carlo returns (bootstrapped only at the rollout end)",
      "with λ = 1 the targets should equal the discounted reward sums, cut at every done")
