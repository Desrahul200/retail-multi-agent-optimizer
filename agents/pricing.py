import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class BetaBandit:
    def __init__(self, price_grid, alpha=None, beta=None):
        self.price_grid = price_grid
        K = len(price_grid)
        self.alpha = alpha if alpha is not None else np.ones(K)
        self.beta  = beta  if beta  is not None else np.ones(K)
    def sample_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        idx = np.argmax(samples)
        return idx, self.price_grid[idx]
    def update(self, idx, successes, trials):
        self.alpha[idx] += successes
        self.beta[idx]  += (trials - successes)

def run(state: dict, full_df: pd.DataFrame) -> dict:
    sub = full_df[full_df["product_id"] == state["product_id"]].copy()
    if sub.empty:
        return state
    # ---------- 1. Fit log-log elasticity model ---------------------------
    tbl = sub[["units_sold", "price", "promotion_flag"]].copy()
    tbl["log_units"] = np.log1p(tbl["units_sold"])
    tbl["log_price"] = np.log(tbl["price"])
    X = tbl[["log_price", "promotion_flag"]]
    y = tbl["log_units"]
    model = LinearRegression().fit(X, y)
    # ---------- 2. Build tight price grid ---------------------------------
    lo = tbl["price"].min() * 0.9
    hi = tbl["price"].max() * 1.1
    price_grid = np.linspace(lo, hi, 25)          # 25 candidate prices
    # ---------- 3. Demand prediction for posterior update -----------------
    X_pred = pd.DataFrame({
        "log_price": np.log(price_grid),
        "promotion_flag": tbl["promotion_flag"].mode()[0]
    })
    demand_pred = np.expm1(model.predict(X_pred))   # vector length = 25
    # ---------- 4. Thompson-sampling bandit -------------------------------
    if "bandit_alpha" in state:
        bandit = BetaBandit(price_grid,
                            alpha=np.array(state["bandit_alpha"]),
                            beta =np.array(state["bandit_beta"]))
    else:
        bandit = BetaBandit(price_grid)              # α=β=1 prior
    # one beta draw per arm ➜ exploration
    samples  = np.random.beta(bandit.alpha, bandit.beta)
    arm_idx  = np.argmax(samples)
    rec_price = price_grid[arm_idx]
    # Treat predicted demand as “trials”, realised sales as “successes”
    trials    = demand_pred.max()
    successes = demand_pred[arm_idx]                 # proxy for sales
    bandit.update(arm_idx, successes, trials)
    # ---------- 5. Write results back to shared state ---------------------
    revenue_curve = price_grid * demand_pred
    state.update({
        "recommended_price": float(rec_price),
        "expected_demand":   float(demand_pred[arm_idx]),
        "expected_revenue":  float(revenue_curve[arm_idx]),
        "revenue_curve":     pd.DataFrame({"price": price_grid,
                                           "revenue": revenue_curve}),
        "bandit_alpha":      bandit.alpha.tolist(),
        "bandit_beta":       bandit.beta.tolist()
    })
    return state 