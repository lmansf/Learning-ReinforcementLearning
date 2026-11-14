running_average = require("running_average")

out = running_average([2, 4, 6])
check(isinstance(out, list) and len(out) == 3, "Returns a list with one entry per input",
      f"got {out!r}")
check(approx(out, [2, 3, 4]), "running_average([2, 4, 6]) == [2, 3, 4]", f"got {out!r}")
check(running_average([]) == [], "Empty input gives an empty list")
check(approx(running_average([5]), [5]), "Single element")
import random
random.seed(1)
xs = [random.random() for _ in range(1000)]
check(approx(running_average(xs)[-1], sum(xs) / len(xs), 1e-9), "Last element is the overall mean")
