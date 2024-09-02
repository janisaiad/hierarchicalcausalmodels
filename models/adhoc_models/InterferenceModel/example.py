from InterferenceModel import InterferenceUnitModel
import numpy as np

model = InterferenceUnitModel(nodes={'U', 'A', 'Y', 'B'},
                              edges={('A', 'Y'), ('U', 'A'), ('A', 'B'), ('B', 'Y'), ('U', 'Y')},
                              prior=lambda: np.random.normal(0, 1),
                              law_a=lambda u: np.random.normal(u, 1),
                              law_b=lambda l: np.random.normal(l[0], sum(abs(j) for j in l[1:])),
                              law_y=lambda u, a, b: np.random.normal(a, abs(u) + abs(b)),
                              sizes=[10, 5])

for k, v in model.sample().items():
    print(k, v)
