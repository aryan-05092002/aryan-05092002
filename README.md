## Hi there 👋

<!--
**aryan-05092002/aryan-05092002** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
"""
Dynamic OLS (DOLS) estimator for y_t = alpha + beta' x_t + (leads/lags of Δx) + error_t
This function:
 - includes levels of X
 - augments with `p` leads and `q` lags of first-differences of X (Stock & Watson DOLS)
 - returns coefficients, HAC (Newey-West) robust cov, t-stats and p-values
Requirements: pandas, numpy, statsmodels, scipy
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy import stats

def _make_lead_lag_diffs(df_x, leads, lags):
    # df_x: DataFrame of X in levels, index aligned with y
    # returns DataFrame with columns like 'dx_var_lead1', 'dx_var_lag1', ...
    dx = df_x.diff().rename(columns=lambda c: f"d{c}")
    cols = []
    out = pd.DataFrame(index=df_x.index)
    # leads: positive k => include dx_{t+1}, dx_{t+2}, ...
    for k in range(1, leads + 1):
        shifted = dx.shift(-k)
        shifted.columns = [f"{col}_lead{k}" for col in dx.columns]
        out = pd.concat([out, shifted], axis=1)
        cols += list(shifted.columns)
    # lags: positive k => include dx_{t-1}, dx_{t-2}, ...
    for k in range(1, lags + 1):
        shifted = dx.shift(k)
        shifted.columns = [f"{col}_lag{k}" for col in dx.columns]
        out = pd.concat([out, shifted], axis=1)
        cols += list(shifted.columns)
    return out

def dynamic_ols(y, X, leads=2, lags=2, add_trend=False, nw_lags=None, dropna=True):
    """
    y: pd.Series (T)
    X: pd.DataFrame (T x k) (levels)
    leads, lags: numbers of leads/lags of ΔX to include
    add_trend: if True include time trend
    nw_lags: number of lags for Newey-West HAC. If None, uses lags.
    dropna: if True drop rows with nan (due to leads/lags)
    Returns: dict with results (params, conf, tstat, pvals, model results object, robust cov)
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)

    T = len(y)
    dols_X = X.copy()
    # leads/lags of differences
    leadlag = _make_lead_lag_diffs(X, leads=leads, lags=lags)
    # assemble regressors: levels of X + lead/lag diffs + intercept/trend
    regressors = pd.concat([dols_X, leadlag], axis=1)
    if add_trend:
        regressors["trend"] = np.arange(len(regressors))

    regressors = sm.add_constant(regressors, has_constant='add')

    if dropna:
        data = pd.concat([y, regressors], axis=1).dropna()
        y2 = data.iloc[:, 0]
        X2 = data.iloc[:, 1:]
    else:
        y2 = y
        X2 = regressors

    model = sm.OLS(y2, X2)
    res = model.fit()
    # Newey-West HAC covariance
    nw_lags_use = nw_lags if nw_lags is not None else max(1, lags)
    robust_cov = cov_hac(res, nlags=nw_lags_use)
    se_robust = np.sqrt(np.diag(robust_cov))
    params = res.params
    tstats = params / se_robust
    pvals = 2 * (1 - stats.t.cdf(np.abs(tstats), df=res.df_resid))

    output = {
        "params": params,
        "robust_cov": robust_cov,
        "robust_se": pd.Series(se_robust, index=params.index),
        "tstats": pd.Series(tstats, index=params.index),
        "pvalues": pd.Series(pvals, index=params.index),
        "nobs": int(res.nobs),
        "df_resid": res.df_resid,
        "ols_result": res,
        "spec": {"leads": leads, "lags": lags, "add_trend": add_trend}
    }
    return output

# -------------------------
# Example: simulate cointegrated data
# -------------------------
if __name__ == "__main__":
    np.random.seed(123)
    T = 300
    # generate x as random walk
    eps_x = np.random.normal(size=T)
    x = np.cumsum(eps_x)
    # y = 1.5 + 2.0*x + stationary noise
    eps_y = np.random.normal(scale=1.0, size=T)
    y = 1.5 + 2.0 * x + eps_y

    idx = pd.RangeIndex(start=0, stop=T, step=1)
    y_s = pd.Series(y, index=idx, name='y')
    X_df = pd.DataFrame({'x': x}, index=idx)

    res = dynamic_ols(y_s, X_df, leads=2, lags=2, add_trend=False, nw_lags=4)
    print("Estimated beta on x (level):")
    print(res["params"]["x"])
    print("\nRobust SE, t-stat and p-value for 'x':")
    print(res["robust_se"]["x"], res["tstats"]["x"], res["pvalues"]["x"])
    print("\nFull parameter table:")
    tab = pd.DataFrame({
        "coef": res["params"],
        "robust_se": res["robust_se"],
        "t": res["tstats"],
        "pvalue": res["pvalues"]
    })
    print(tab)
