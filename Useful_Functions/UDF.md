# User defined functions

## Python Structure for object oriented programming

```py
#!/usr/bin/python

from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


class Employee:
    'Common base class for all employees'
    empCount = 0

    def __init__(self, name, salary):
      self.name = name
      self.salary = salary
      Employee.empCount += 1

    def displayCount(self):
      print "Total Employee %d" % Employee.empCount

    def displayEmployee(self):
      print "Name : ", self.name,  ", Salary: ", self.salary


def main(debug = False):
  "This would create first object of Employee class"
  emp1 = Employee("Zara", 2000)
  "This would create second object of Employee class"
  emp2 = Employee("Manni", 5000)
  emp1.displayEmployee()
  emp2.displayEmployee()
  print "Total Employee %d" % Employee.empCount

if __name__ == "__main__":
  with timer("Full run"):
    main()

```

## Import sniplets

#### Python Basic

```py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

#### Data Manipulation

```py
import numpy as np
import pandas as pd
```
#### Data Visualization

```py
import matplotlib.pyplot as plt
import seaborn as sns
```

#### Machine Learning

```py

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

from scipy.stats import norm
from scipy.optimize import curve_fit
```

## UDF for python utils

```py
from contextlib import contextmanager

@contextmanager
def timer(title):
  '''Example Google Style Python Docstrings

  Args:
       title (str): Label to identify the call

  Returns:
       None

  Example:
    with timer("Full run"):
      main()

  .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html

  '''

    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

```

## UDF for Data Preparation

### One-hot encoding for categorical columns with get_dummies

```py
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
```





## UDF Dump

```py
def gauss(x, mu, sigma, A):
    return A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)
```
