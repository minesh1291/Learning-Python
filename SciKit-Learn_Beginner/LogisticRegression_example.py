#One of the simplest options to get a feeling for the "influence" of a given parameter in a linear classification model (logistic being one of those), is to consider the magnitude of its coefficient times the standard deviation of the corresponding parameter in the data.

#Consider this example:

import numpy as np    
from sklearn.linear_model import LogisticRegression

x1 = np.random.randn(100)
x2 = 4*np.random.randn(100)
x3 = 0.5*np.random.randn(100)
y = (3 + x1 + x2 + x3 + 0.2*np.random.randn()) > 0
X = np.column_stack([x1, x2, x3])

m = LogisticRegression()
m.fit(X, y)

# The estimated coefficients will all be around 1:
print(m.coef_)

# Those values, however, will show that the second parameter
# is more influential
print(np.std(X, 0)*m.coef_)

#An alternative way to get a similar result is to examine the coefficients of the model fit on standardized parameters:

m.fit(X / np.std(X, 0), y)
print(m.coef_)

#Note that this is the most basic approach and a number of other techniques for finding feature importance or parameter influence exist (using p-values, bootstrap scores, various "discriminative indices", etc).

# ref. https://stackoverflow.com/questions/34052115/how-to-find-the-importance-of-the-features-for-a-logistic-regression-model

