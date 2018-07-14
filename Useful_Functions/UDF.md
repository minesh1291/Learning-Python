<!-- TOC START min:1 max:3 link:true update:true -->
- [User defined functions [Read to Use]](#user-defined-functions-read-to-use)
  - [Python Structure for object oriented programming](#python-structure-for-object-oriented-programming)
  - [Import sniplets](#import-sniplets)
    - [Python Basic](#python-basic)
    - [Data Mining](#data-mining)
    - [Data Manipulation](#data-manipulation)
    - [Data Visualization](#data-visualization)
    - [Machine Learning / Statistical Modelling](#machine-learning--statistical-modelling)
  - [UDF for Python Utils](#udf-for-python-utils)
    - [Task timer](#task-timer)
  - [UDF for Data Preparation](#udf-for-data-preparation)
    - [One-hot encoding for categorical columns with get_dummies](#one-hot-encoding-for-categorical-columns-with-get_dummies)
  - [UDF for Model Selection](#udf-for-model-selection)
    - [LightGBM GBDT with KFold or Stratified KFold AND Display/plot feature Importance](#lightgbm-gbdt-with-kfold-or-stratified-kfold-and-displayplot-feature-importance)
    - [Plot Performance](#plot-performance)
  - [UDF Dump](#udf-dump)

<!-- TOC END -->

# User defined functions [Read to Use]

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

### Python Basic

```py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```
### Data Mining

```py
import json
```

### Data Manipulation

```py
import numpy as np
import pandas as pd
```
### Data Visualization

```py
import matplotlib.pyplot as plt
import seaborn as sns
```

### Machine Learning / Statistical Modelling

```py

import lightgbm as lgb
import xgboost as xgb
# Alternative
from lightgbm import LGBMClassifier

import sklearn
# Alternative
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from scipy.stats import norm
from scipy.optimize import curve_fit
```

## UDF for Python Utils
### Task timer
```py
from contextlib import contextmanager
import time

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

## UDF for Model Selection

### LightGBM GBDT with KFold or Stratified KFold AND Display/plot feature Importance

```py
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by <xyz> optimization
        clf = LGBMClassifier(
            nthread=4,
            #is_unbalance=True,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=32,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.04,
            reg_lambda=0.073,
            min_split_gain=0.0222415,
            min_child_weight=40,
            silent=-1,
            verbose=-1,
            #scale_pos_weight=11
            )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric= 'auc', verbose= 1000, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

```

### Plot Performance

```py
def display_roc_curve(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6, 6))
    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    plt.plot(
        [0, 1], [0, 1],
        linestyle='--',
        lw=2,
        color='r',
        label='Luck',
        alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(
        fpr,
        tpr,
        color='b',
        label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
        lw=2,
        alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig('roc_curve-01.png')


def display_precision_recall(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6, 6))

    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(
        precision,
        recall,
        color='b',
        label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
        lw=2,
        alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('LightGBM Recall / Precision')
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig('recall_precision_curve-01.png')

```


## UDF Dump

```py
def gauss(x, mu, sigma, A):
    return A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)
```
