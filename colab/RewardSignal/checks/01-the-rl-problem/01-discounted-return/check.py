discounted_return = require("discounted_return")
returns_to_go = require("returns_to_go")

g = discounted_return([1, 1, 1], 0.5)
check(approx(g, 1.75), "discounted_return([1,1,1], 0.5) == 1.75", f"got {g}")
g = discounted_return([0, 0, 10], 0.9)
check(approx(g, 8.1), "A delayed reward is discounted twice (γ²·10 = 8.1)", f"got {g}")
check(approx(discounted_return([3, 4], 0.0), 3), "γ = 0 only counts the first reward")
check(approx(discounted_return([], 0.9), 0), "No rewards → return 0")

rtg = returns_to_go([1, 2, 3], 0.5)
check(isinstance(rtg, (list, tuple)) or hasattr(rtg, "__len__"), "returns_to_go returns a sequence")
check(approx(list(rtg), [2.75, 3.5, 3.0]), "returns_to_go([1,2,3], 0.5) == [2.75, 3.5, 3.0]",
      f"got {list(rtg)} — remember G_T = 0 after the last reward, and walk backwards")

import random
random.seed(3)
rs = [random.uniform(-1, 1) for _ in range(50)]
rtg = list(returns_to_go(rs, 0.97))
check(approx(rtg[0], discounted_return(rs, 0.97), 1e-9) and approx(rtg[10], discounted_return(rs[10:], 0.97), 1e-9),
      "returns_to_go agrees with discounted_return at every step")
