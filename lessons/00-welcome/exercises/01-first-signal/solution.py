import random
import rlhud


def running_average(xs):
    """Return a list whose i-th element is the mean of xs[0], ..., xs[i]."""
    out, total = [], 0.0
    for i, x in enumerate(xs):
        total += x
        out.append(total / (i + 1))
    return out


if __name__ == "__main__":
    # A noisy signal around 1.0 — watch the running average settle.
    random.seed(0)
    noisy = [random.gauss(1.0, 2.0) for _ in range(300)]
    rlhud.plot(noisy, label="raw reward", title="Signal acquisition")
    rlhud.plot(running_average(noisy), label="running average", title="Signal acquisition")
    rlhud.stat("Final estimate", running_average(noisy)[-1])
