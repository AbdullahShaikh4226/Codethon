"""The three candidate models. Each predicts log(minutes) and returns minutes."""
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBRegressor

ALGOS = ("Gradient Boosting", "LightGBM", "CatBoost", "XGBoost")

# Small grids so each library gets a fair shot; the best setting per platform is picked by CV MAE.
GRIDS = {
    "Gradient Boosting": [
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 800, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.1},
        {"n_estimators": 800, "max_depth": 5, "learning_rate": 0.05},
    ],
    "LightGBM": [
        {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 5},
        {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 20},
        {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 5},
        {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20},
    ],
    "CatBoost": [
        {"iterations": 1000, "depth": 4, "learning_rate": 0.05},
        {"iterations": 1000, "depth": 4, "learning_rate": 0.1},
        {"iterations": 1000, "depth": 6, "learning_rate": 0.05},
        {"iterations": 1000, "depth": 6, "learning_rate": 0.1},
    ],
    "XGBoost": [
        {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.1},
        {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.05},
        {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.1},
    ],
}


def _to_category(X, cols):
    # LightGBM splits natively on pandas 'category' columns and remembers the training categories.
    return X.astype({c: "category" for c in cols})


# LightGBM's categorical defaults (min_data_per_group=100, cat_smooth=10, cat_l2=10) suit big data; on
# 1,000 rows they blocked useful splits: AWS CV MAE 17.8 min with defaults vs 2.6 min with these.
LGBM_SMALL_DATA = {"min_data_per_group": 1, "cat_smooth": 1, "cat_l2": 1}


# Every category here has at most 4 values, so one-hot them instead of building target statistics:
# 16 s → 0.4 s per fit, and lower CV MAE on all three platforms (VMware 28.0 → 22.2 min).
CATBOOST_SMALL_CATS = {"one_hot_max_size": 10}


class CatBoostModel(RegressorMixin, BaseEstimator):
    """CatBoost behind a clone-safe wrapper: CatBoostRegressor rewrites its cat_features argument,
    which scikit-learn 1.8's clone() rejects."""

    def __init__(self, cat=(), params=None):
        self.cat = cat
        self.params = params

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(cat_features=list(self.cat), random_seed=0, verbose=0,
                                        allow_writing_files=False, **CATBOOST_SMALL_CATS, **(self.params or {}))
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def build(algo: str, cat: list, params: dict):
    """Estimator taking the raw feature DataFrame (numeric + text categories) and predicting minutes."""
    if algo == "Gradient Boosting":
        est = Pipeline([
            ("onehot", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
                                         remainder="passthrough", verbose_feature_names_out=False)),
            ("model", GradientBoostingRegressor(random_state=0, **params)),
        ])
    elif algo == "LightGBM":
        est = Pipeline([
            ("category", FunctionTransformer(_to_category, kw_args={"cols": cat})),
            ("model", LGBMRegressor(random_state=0, verbose=-1, **LGBM_SMALL_DATA, **params)),
        ])
    elif algo == "CatBoost":
        est = CatBoostModel(cat=tuple(cat), params=params)
    elif algo == "XGBoost":
        est = Pipeline([
            ("onehot", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
                                         remainder="passthrough", verbose_feature_names_out=False)),
            ("model", XGBRegressor(random_state=0, tree_method="hist", n_jobs=-1, **params)),
        ])
    else:
        raise ValueError(f"Unknown algorithm {algo!r}")
    # Restore times are heavily skewed (VMware 6 to 11,930 min); learning log(minutes) treats a 10% miss
    # the same at every size.
    return TransformedTargetRegressor(regressor=est, func=np.log, inverse_func=np.exp)
