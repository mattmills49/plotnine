import numpy as np
import pandas as pd

from .._utils import resolution
from ..doctools import document
from .stat import stat


@document
class stat_bardist(stat):
    """
    Compute bardist statistics

    {usage}

    Parameters
    ----------
    {common_parameters}
    intervals : collection, default=[.5, .8, .95]
        Intervals to plot

    See Also
    --------
    plotnine.geoms.geom_bardist
    """

    _aesthetics_doc = """
    {aesthetics_table}

    **Options for computed aesthetics**

    ```python
    "lower_small" # lower limit of 50% interval, 25th quantile
    "upper_small" # upper limit of 50% interval, 75th quantile
    "lower_mid" # lower limit of 80% interval, 10th quantile
    "upper_mid" # upper limit of 80% interval, 90th quantile
    "lower_big" # lower limit of 95% interval, 2.5th quantile
    "upper_big" # upper limit of 95% interval, 97.5th quantile
    ```
    
       'n'     # Number of observations at a position
       
    Calculated aesthetics are accessed using the `after_stat` function.
    e.g. `after_stat('width')`{.py}.
    """

    REQUIRED_AES = {"x", "y"}
    NON_MISSING_AES = {"weight"}
    DEFAULT_PARAMS = {
        "geom": "bardist",
        "position": "dodge",
        "na_rm": True,
        "intervals": [0.5, 0.8, 0.95]
    }
    CREATES = {
        "lower_small",
        "upper_small",
        "lower_mid",
        "upper_mid",
        "lower_big",
        "upper_big",
        "n"
    }

    def setup_data(self, data):
        if "x" not in data:
            data["x"] = 0
        return data

    def setup_params(self, data):
        ## We need to convert the intervals to the actual quantiles we plot    
        qs = list()
        self.params["intervals"] = sorted(self.params["intervals"])
        for i in self.params["intervals"]:
            qs += [1 - i/2, 0.5 + i/2]
            
        self.params["qs"] = qs
        return self.params

    @classmethod
    def compute_group(cls, data, scales, **params):
        n = len(data)
        y = data["y"].to_numpy()
        if "weight" in data:
            weights = data["weight"]
            total_weight = np.sum(weights)
        else:
            weights = None
            total_weight = len(y)
        res = weighted_percentile(y, q = params["qs"], weights=weights)

        if isinstance(data["x"].dtype, pd.CategoricalDtype):
            x = data["x"].iloc[0]
        else:
            x = np.mean([data["x"].min(), data["x"].max()])

        d = {
            "lower_small": res[0],
            "upper_small": res[1],
            "lower_mid": res[2],
            "upper_mid": res[3],
            "lower_big": res[4],
            "upper_big": res[5],
            "n": n,
            "x": x
        }
        return pd.DataFrame(d)


def weighted_percentile(a, q, weights=None):
    """
    Compute the weighted q-th percentile of data

    Parameters
    ----------
    a : array_like
        Input that can be converted into an array.
    q : array_like[float]
        Percentile or sequence of percentiles to compute. Must be int
        the range [0, 100]
    weights : array_like
        Weights associated with the input values.
    """
    # Calculate and interpolate weighted percentiles
    # method derived from https://en.wikipedia.org/wiki/Percentile
    # using numpy's standard C = 1
    if weights is None:
        weights = np.ones(len(a))

    weights = np.asarray(weights)
    q = np.asarray(q)

    C = 1
    idx_s = np.argsort(a)
    a_s = a[idx_s]
    w_n = weights[idx_s]
    S_N = np.sum(weights)
    S_n = np.cumsum(w_n)
    p_n = (S_n - C * w_n) / (S_N + (1 - 2 * C) * w_n)
    pcts = np.interp(q / 100.0, p_n, a_s)
    return pcts



