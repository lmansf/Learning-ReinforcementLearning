import numpy as np

ReplayBuffer = require("ReplayBuffer")
huber = require("huber")
huber_grad = require("huber_grad")

# --- Huber first (independent of the buffer)
h = np.asarray(huber(np.array([0.0, 0.5, -0.5, 2.0, -3.0]), 1.0))
check(approx(h, [0, 0.125, 0.125, 1.5, 2.5]), "huber: ½x² inside, δ(|x| − ½δ) outside",
      f"for x = [0, .5, −.5, 2, −3] expected [0, .125, .125, 1.5, 2.5], got {h.tolist()}")
h = np.asarray(huber(np.array([3.0]), delta=2.0))
check(approx(h, [4.0]), "huber respects delta", f"huber(3, δ=2) = 2·(3 − 1) = 4, got {h.tolist()}")
g = np.asarray(huber_grad(np.array([0.3, -0.7, 5.0, -2.0]), 1.0))
check(approx(g, [0.3, -0.7, 1.0, -1.0]), "huber_grad = clip(x, −δ, δ)", f"got {g.tolist()}")

# --- buffer basics
buf = ReplayBuffer(capacity=3, obs_dim=2, seed=0)
check(len(buf) == 0, "A new buffer is empty", f"len = {len(buf)}")
for i in range(2):
    buf.add([i, i], i % 2, float(i), [i + 1, i + 1], False)
check(len(buf) == 2, "len grows with add()", f"len = {len(buf)}")
S, A, R, S2, D = buf.sample(200)
check(np.shape(S) == (200, 2) and np.shape(A) == (200,) and np.shape(D) == (200,),
      "sample returns batch arrays: S (B, obs_dim), A/R/D (B,)",
      f"shapes S{np.shape(S)} A{np.shape(A)} R{np.shape(R)} S2{np.shape(S2)} D{np.shape(D)}")
check(set(np.asarray(R).tolist()) == {0.0, 1.0}, "Samples come only from filled slots",
      f"rewards seen: {sorted(set(np.asarray(R).tolist()))} — sample from [0, size), not [0, capacity)")
check(np.all(np.asarray(S2)[:, 0] == np.asarray(S)[:, 0] + 1), "Each sample keeps (s, a, r, s′) together",
      "s′ should be the next state of the *same* transition — use one index array for all fields")

for i in range(2, 5):
    buf.add([i, i], i % 2, float(i), [i + 1, i + 1], i == 4)
check(len(buf) == 3, "len never exceeds capacity", f"len = {len(buf)}")
S, A, R, S2, D = buf.sample(300)
seen = sorted(set(np.asarray(R).tolist()))
check(seen == [2.0, 3.0, 4.0], "When full, the oldest transitions are overwritten (circular buffer)",
      f"after adding rewards 0..4 to a capacity-3 buffer you should hold 2, 3, 4 — got {seen}")
D = np.asarray(D, dtype=float); R = np.asarray(R)
check(np.all(D[R == 4.0] == 1.0) and np.all(D[R != 4.0] == 0.0), "terminated flags are stored per transition")

# --- uniformity on a bigger buffer
buf = ReplayBuffer(capacity=1000, obs_dim=1, seed=1)
for i in range(1000):
    buf.add([i], 0, float(i), [i], False)
R = np.asarray(buf.sample(20000)[2])
counts = np.histogram(R, bins=10, range=(0, 1000))[0]
check(counts.min() > 1700 and counts.max() < 2300, "Sampling is uniform over stored transitions",
      f"counts per tenth of the buffer: {counts.tolist()}")
