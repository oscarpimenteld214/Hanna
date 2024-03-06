
from typing import Tuple, List
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp, norm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb

import functions as func

pd.options.mode.copy_on_write = True
pd.set_option("mode.copy_on_write", False)

# ------------------------------
# 3. Univariate Analysis
# ------------------------------

# 1. Long list of variables

woe, iv = func.woe_iv(train, "GB", bins=5)
woe.to_csv("tables/woe_test.csv")
iv.to_csv("tables/iv_test.csv")

top20_variables = iv.sort_values(ascending=False).head(20).index

# 2. Visualize the WOE

for var in top20_variables:
    func.woe_plot(woe, var)
