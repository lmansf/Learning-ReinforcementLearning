import rlhud


def discounted_return(rewards, gamma):
    """G0 = r1 + gamma*r2 + gamma^2*r3 + ...   (rewards = [r1, r2, r3, ...])"""
    return sum((gamma ** k) * r for k, r in enumerate(rewards))


def returns_to_go(rewards, gamma):
    """[G0, G1, ..., G_{T-1}] computed backwards with G_t = r_{t+1} + gamma * G_{t+1}."""
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return out[::-1]


if __name__ == "__main__":
    rewards = [0, 0, 0, 1, 0, 0, 0, 0, 0, 10]
    for gamma in (0.0, 0.5, 0.9, 0.99):
        print(f"gamma={gamma:<5} G0={discounted_return(rewards, gamma):8.4f}")
        rlhud.plot(returns_to_go(rewards, gamma), label=f"γ={gamma}", title="Return-to-go per step")
