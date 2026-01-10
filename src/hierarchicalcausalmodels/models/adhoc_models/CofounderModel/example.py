from CofounderModel import CofounderUnitModel
import numpy as np

model = CofounderUnitModel(nodes={'U', 'A', 'Y'},
                           edges={('A', 'Y'), ('U', 'A'), ('U', 'Y')},
                           prior=lambda: np.random.normal(0, 1),
                           law_a=lambda u: np.random.normal(u, 1),
                           law_y=lambda u, a: np.random.normal(a, abs(u)),
                           sizes=[10, 5])

for k, v in model.sample().items():
    print(k, v)

