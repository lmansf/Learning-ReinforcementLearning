import numpy as np
from inventory import InventoryEnv, s_S_policy, evaluate_policy

context = require("context")
adjusted_action = require("adjusted_action")
train_residual_q = require("train_residual_q")
residual_policy = require("residual_policy")
OFFS = np.array([-20, -10, 0, 10, 20, 30, 40])
BASE = (50, 70)

# ---- 1. context -----------------------------------------------------------------------------
env = InventoryEnv()
env.reset(seed=3)
ok = True
for s in (0, 1, 2, 13, 27, 200):
    b, dow, promo = env.decode_state(s)
    if context(env, s) != dow * 2 + promo:
        ok = False
        check(False, "context(env, obs) = dow * 2 + promo_soon", f"state {s} decodes to dow={dow}, promo_soon={promo}; you returned {context(env, s)}")
        break
if ok:
    check(True, "context(env, obs) = dow * 2 + promo_soon")

# ---- 2. adjusted_action ---------------------------------------------------------------------
env.on_hand, env.pipeline, env._transit = 30, [0, 12], 12        # position 42, order step 6
check(adjusted_action(env, 50, 70, 0) == env.nearest_action(28), "position 42 ≤ s=50 → order up to 70 (28 units)",
      f"expected action {env.nearest_action(28)}, got {adjusted_action(env, 50, 70, 0)}")
check(adjusted_action(env, 50, 70, -10) == 0, "offset −10 lowers s to 40, so at position 42 we order nothing",
      f"got action {adjusted_action(env, 50, 70, -10)}")
check(adjusted_action(env, 50, 70, 20) == env.nearest_action(48), "offset +20 → order up to 90 (48 units)",
      f"expected {env.nearest_action(48)}, got {adjusted_action(env, 50, 70, 20)}")

# ---- 3. a short training run ----------------------------------------------------------------
short = InventoryEnv(horizon=30)
Q, rets = train_residual_q(short, *BASE, offsets=OFFS, episodes=20, seed=0)
Q = np.asarray(Q)
check(Q.shape == (14, len(OFFS)), "Q has shape (14 contexts, len(offsets))", f"got {Q.shape}")
check(len(rets) == 20 and all(np.isfinite(rets)), "one finite profit per training episode", f"got {len(rets)} values")
Q0, r0 = train_residual_q(short, *BASE, offsets=OFFS, episodes=0, seed=0)
Q0 = np.asarray(Q0)
check(Q0.shape == (14, len(OFFS)) and np.all(np.argmax(Q0, axis=1) == 2),
      "before any training the greedy choice is offset 0 everywhere (the agent starts as the planner)",
      f"untrained Q rows should all prefer offsets[2] = 0; argmax per row = {np.argmax(Q0, axis=1) if Q0.ndim == 2 else Q0}")
Qp = np.zeros((14, len(OFFS))); Qp[:, 2] = 1.0
pol = residual_policy(short, Qp, *BASE, offsets=OFFS)
plan = s_S_policy(short, *BASE)
obs, _ = short.reset(seed=5)
same = True
for _ in range(30):
    a, b = pol(obs), plan(obs)
    same &= (a == b)
    obs, *_ = short.step(b)
check(same, "with all preference on offset 0, residual_policy behaves exactly like the planner's (s, S)",
      "residual_policy(env, Q, s, S, offsets) should pick argmax_k Q[context] and call adjusted_action")

# ---- 4. beat the planner on held-out years --------------------------------------------------
env = InventoryEnv()
TEST = range(100, 120)
base = evaluate_policy(env, lambda e: s_S_policy(e, *BASE), seeds=TEST)
ratios = []
for seed in (0, 1, 2):
    Q, _ = train_residual_q(env, *BASE, offsets=OFFS, seed=seed)
    ratios.append(evaluate_policy(env, lambda e: residual_policy(e, Q, *BASE, offsets=OFFS), seeds=TEST) / base)
mean = float(np.mean(ratios))
check(mean >= 1.02, "Residual Q beats the tuned planner by ≥ 2% on 20 held-out years (mean of 3 training seeds)",
      f"profit ratios vs planner: {np.round(ratios, 3)} (mean {mean:.3f}). Check the ε schedule (decay to 0 by 70% "
      "of training), the +1 head start on offset 0, and that the target uses max over the NEXT context.")
