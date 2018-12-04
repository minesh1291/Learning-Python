# how to set dtype for nested numpy ndarray?
```py
import numpy as np

data = [('spire', '250um', [(0, 1.89e6, 0.0), (1, 2e6, 1e-2)]),
        ('spire', '350',   [(0, 1.89e6, 0.0), (2, 2.02e6, 3.8e-2)])
        ]
table = np.array(data, dtype=[('instrument', '|S32'),
                               ('filter', '|S64'),
                               ('response', [('linenumber', 'i'),
                                             ('wavelength', 'f'),
                                             ('throughput', 'f')], (2,))
                              ])

print table[0]
# gives ('spire', '250um', [(0, 1890000.0, 0.0), (1, 2000000.0, 0.009999999776482582)])
```
