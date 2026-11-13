import numpy as np

ppo_clip_loss = require("ppo_clip_loss")
ppo_diagnostics = require("ppo_diagnostics")

L = np.log
# ratio = 1 everywhere → loss = −mean(adv), grad = −adv/N
loss, g = ppo_clip_loss(L([0.5, 0.5]), L([0.5, 0.5]), np.array([2.0, -1.0]), 0.2)
check(approx(loss, -0.5), "ratio = 1: loss = −mean(Â)", f"expected −0.5, got {loss}")
check(approx(g, [-1.0, 0.5]), "ratio = 1: gradient = −Â/N", f"expected [−1, 0.5], got {np.asarray(g).tolist()}")

# One sample per case, ε = 0.2. logp_old = log 0.5.
cases = [
    # (π_new, Â, expected loss, expected grad, description)
    (0.55, +1.0, -1.1, -1.1, "Â > 0, r = 1.1 (inside the clip range): normal policy-gradient step"),
    (0.75, +1.0, -1.2, 0.0, "Â > 0, r = 1.5: objective clipped at 1+ε, gradient 0 (don't push further)"),
    (0.25, +1.0, -0.5, -0.5, "Â > 0, r = 0.5: NOT clipped — pessimistic min keeps the gradient to undo the mistake"),
    (0.45, -1.0, 0.9, 0.9, "Â < 0, r = 0.9 (inside the range): normal step"),
    (0.25, -1.0, 0.8, 0.0, "Â < 0, r = 0.5: objective clipped at 1−ε, gradient 0"),
    (0.75, -1.0, 1.5, 1.5, "Â < 0, r = 1.5: NOT clipped — the bad action became likelier, keep pushing it down"),
]
for p_new, A, exp_loss, exp_g, label in cases:
    loss, g = ppo_clip_loss(L([p_new]), L([0.5]), np.array([A]), 0.2)
    check(approx(loss, exp_loss) and approx(g, [exp_g]), label,
          f"expected loss {exp_loss}, grad {exp_g}; got loss {loss:.4f}, grad {np.asarray(g).tolist()}")

# gradient matches finite differences away from the kinks
rng = np.random.default_rng(0)
lo = L(rng.uniform(0.2, 0.8, 50)); ln = lo + rng.normal(0, 0.3, 50); A = rng.normal(size=50)
r = np.exp(ln - lo)
ok = (np.abs(r - 0.8) > 1e-3) & (np.abs(r - 1.2) > 1e-3)
_, g = ppo_clip_loss(ln, lo, A, 0.2)
num = np.zeros(50)
for i in range(50):
    e = np.zeros(50); e[i] = 1e-6
    num[i] = (ppo_clip_loss(ln + e, lo, A, 0.2)[0] - ppo_clip_loss(ln - e, lo, A, 0.2)[0]) / 2e-6
check(approx(np.asarray(g)[ok], num[ok], 1e-6), "Gradient matches finite differences on a random batch")

kl, cf = ppo_diagnostics(L([0.5, 0.5, 0.5]), L([0.5, 0.5, 0.5]))
check(approx(kl, 0) and approx(cf, 0), "Identical policies: KL 0, clip fraction 0", f"got {kl}, {cf}")
kl, cf = ppo_diagnostics(L([0.75, 0.5, 0.25, 0.55]), L([0.5, 0.5, 0.5, 0.5]), 0.2)
exp_kl = np.mean([(r - 1) - np.log(r) for r in (1.5, 1.0, 0.5, 1.1)])
check(approx(kl, exp_kl) and approx(cf, 0.5), "approx_kl = mean((r−1) − log r), clip_fraction = mean(|r−1| > ε)",
      f"expected KL {exp_kl:.4f} and clip fraction 0.5, got {kl}, {cf}")
