#> Every lesson's code starts the same way: `numpy` for arrays and maths, `rlhud` for visuals.
import numpy as np
import rlhud
#> `rlenvs` holds the course's small environments. `GridWorld` builds a maze from an ASCII map.
from rlenvs import GridWorld


#> A *policy* is just a function: observation in, action out. Here: always go RIGHT (action 1).
def always_right(obs):
    #> `RIGHT` is 1 in GridWorld (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT). Returning a plain int is all a policy must do.
    return 1


#> A `rollout` plays one episode and records every reward. You will write this loop in almost every lesson.
def rollout(env, policy, seed=None):
    #> `reset` starts a new episode and returns `(observation, info)` — unpack both, even if you ignore `info`.
    obs, info = env.reset(seed=seed)
    #> Collect rewards in a plain Python list; `list.append` is cheap.
    rewards = []
    #> `while True` + `break` is the idiomatic loop when you don't know the episode length in advance.
    while True:
        #> Ask the policy what to do in the current state.
        action = policy(obs)
        #> `step` returns 5 values. `terminated` = the episode really ended; `truncated` = we hit a time limit.
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        #> Stop on either flag — `or` short-circuits, so this is a cheap test.
        if terminated or truncated:
            break
    return rewards


#> Discounted return, written with NumPy instead of a loop: build the weights γ⁰, γ¹, γ², … in one go.
def discounted(rewards, gamma):
    #> `np.asarray` turns the list into an array without copying if it already is one.
    r = np.asarray(rewards, dtype=float)
    #> `np.arange(n)` → [0, 1, …, n-1]; raising γ to that array gives every power at once (broadcasting).
    weights = gamma ** np.arange(len(r))
    #> Element-wise multiply, then sum. `float(...)` converts the NumPy scalar to a plain Python number.
    return float(np.sum(weights * r))


#> Everything under this guard runs when you press RUN, but not when CHECK imports your functions.
if __name__ == "__main__":
    #> Three rows: walk right along the top row to reach the goal `G`.
    env = GridWorld(["S..G", ".#..", "...."], step_reward=-0.1)
    rewards = rollout(env, always_right)
    #> f-strings: `{expression:format}` — `.3f` means three decimals.
    print(f"steps={len(rewards)}  rewards={rewards}  G0={discounted(rewards, 0.9):.3f}")
    #> `rlhud.bars` draws a bar chart in the VISUALS tab.
    rlhud.bars(rewards, labels=[f"R{k + 1}" for k in range(len(rewards))], title="Rewards along the rollout")
    #> A list comprehension: build a list from an expression and a `for` loop in one line.
    curve = [discounted(rewards, g) for g in np.linspace(0, 1, 21)]
    rlhud.plot(curve, x=list(np.linspace(0, 1, 21)), label="G0", title="Return vs γ")
