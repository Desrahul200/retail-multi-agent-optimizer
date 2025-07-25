import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

# ------------------------------------------------------------------
# CONFIG – tweak these if you like
# ------------------------------------------------------------------
CSV       = "dh_demand.csv"      # or synthetic_demand.csv
SKU       = 827656  # no quotes, so it's an int
HOLD_OUT  = 4                    # weeks (if weekly grain) or days if daily

# ------------------------------------------------------------------
# 1) Load & filter
# ------------------------------------------------------------------
# Always treat product_id as str for robust filtering
SKU = str(SKU)
df = pd.read_csv(CSV, parse_dates=["date"])
df["product_id"] = df["product_id"].astype(str)
df["competitor_gap"] = df["price"] - df["competitor_price"]

# Prophet expects cols ds / y and any regressors
prep = df.rename(columns={"date":"ds", "units_sold":"y", "promotion_flag":"promotion_flag", "competitor_gap":"competitor_gap"})[["ds","y","promotion_flag","competitor_gap"]]

# ------------------------------------------------------------------
# 2) Train-test split
# ------------------------------------------------------------------
train = prep.iloc[:-HOLD_OUT].copy()
test  = prep.iloc[-HOLD_OUT:].copy()

print(f"Train rows: {len(train)}, Test rows: {len(test)}")
if len(train) < 2:
    print(f"Not enough data to train Prophet for SKU {SKU}.")
    exit()

results = []

# --- Naive Baseline ---
naive_pred = test["y"].shift(1).bfill()
naive_mape = np.mean(np.abs((test["y"].values - naive_pred.values) / test["y"].values)) * 100
results.append(("Naive (LOCF)", naive_mape))

# --- Prophet Variants ---
def eval_prophet(train, test, smoothing=False, log=False, promo=False, changepoint=None):
    t = train.copy()
    tst = test.copy()
    label = "Prophet"
    if smoothing:
        t["y"] = t["y"].rolling(4, center=True).median()
        label += "+Smooth"
    if log:
        t["y"] = np.log1p(t["y"])
        label += "+Log"
    if changepoint is not None:
        m = Prophet(yearly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=changepoint)  # type: ignore
        label += f"+CP={changepoint}"
    else:
        m = Prophet(yearly_seasonality=True, daily_seasonality=False)  # type: ignore
    if promo:
        m.add_regressor("promotion_flag")
        label += "+Promo"
    # Fit
    m.fit(t)
    # Forecast
    if promo:
        future = tst[["ds", "promotion_flag"]].copy()
    else:
        future = tst[["ds"]].copy()
    fc = m.predict(future)
    if log:
        yhat = np.expm1(fc["yhat"])
    else:
        yhat = fc["yhat"]
    y_true = tst["y"].values
    mape = np.mean(np.abs((y_true - yhat) / y_true)) * 100
    return label, mape

# Vanilla Prophet
results.append(eval_prophet(train, test))
# Smoothing only
results.append(eval_prophet(train, test, smoothing=True))
# Log only
results.append(eval_prophet(train, test, log=True))
# Promo only
results.append(eval_prophet(train, test, promo=True))
# Changepoint only
results.append(eval_prophet(train, test, changepoint=0.01))
# Smoothing + Log
results.append(eval_prophet(train, test, smoothing=True, log=True))
# Log + Promo
results.append(eval_prophet(train, test, log=True, promo=True))
# Log + Promo + Changepoint
results.append(eval_prophet(train, test, log=True, promo=True, changepoint=0.01))

print(f"\nProphet evaluation for SKU {SKU}  |  hold-out = {HOLD_OUT} weeks")
print(f"{'Variant':<30}  {'MAPE (%)':>10}")
print("-"*45)
for label, mape in results:
    print(f"{label:<30}  {mape:10.2f}") 

# --- LightGBM ---
features = ["promotion_flag", "competitor_gap"]
lgb_train = train.dropna(subset=features + ["y"])
lgb_test = test.dropna(subset=features)
model = lgb.LGBMRegressor()
model.fit(lgb_train[features], lgb_train["y"])
yhat_lgb = model.predict(lgb_test[features])
lgb_mape = np.mean(np.abs((lgb_test["y"].values - yhat_lgb) / lgb_test["y"].values)) * 100
print(f"LightGBM MAPE: {lgb_mape:.2f} %")

# --- Batch Processing for All SKUs ---
CSV = "dh_demand.csv"
HOLD_OUT = 4

df = pd.read_csv(CSV, parse_dates=["date"])
df["product_id"] = df["product_id"].astype(str)
df["competitor_gap"] = df["price"] - df["competitor_price"]

results = []

for SKU in df["product_id"].unique():
    sku_df = df[df["product_id"] == SKU].sort_values("date")
    prep = sku_df.rename(columns={"date":"ds", "units_sold":"y", "promotion_flag":"promotion_flag", "competitor_gap":"competitor_gap"})[["ds","y","promotion_flag","competitor_gap"]]
    if len(prep) <= HOLD_OUT + 2:
        continue
    train = prep.iloc[:-HOLD_OUT].copy()
    test  = prep.iloc[-HOLD_OUT:].copy()
    # Outlier trimming
    lower = train["y"].quantile(0.05)
    upper = train["y"].quantile(0.95)
    train["y"] = train["y"].clip(lower=lower, upper=upper)
    # Naive
    naive_pred = test["y"].shift(1).bfill()
    naive_mape = np.mean(np.abs((test["y"].values - naive_pred.values) / test["y"].values)) * 100
    # Prophet with competitor_gap
    m = Prophet(yearly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.2)  # type: ignore
    m.add_regressor("promotion_flag")
    m.add_regressor("competitor_gap")
    m.fit(train)
    future = test[["ds", "promotion_flag", "competitor_gap"]].copy()
    fc = m.predict(future)
    yhat = fc["yhat"]
    prophet_mape = np.mean(np.abs((test["y"].values - yhat) / test["y"].values)) * 100
    # LightGBM
    features = ["promotion_flag", "competitor_gap"]
    lgb_train = train.dropna(subset=features + ["y"])
    lgb_test = test.dropna(subset=features)
    model = lgb.LGBMRegressor()
    model.fit(lgb_train[features], lgb_train["y"])
    yhat_lgb = model.predict(lgb_test[features])
    lgb_mape = np.mean(np.abs((lgb_test["y"].values - yhat_lgb) / lgb_test["y"].values)) * 100
    # Store results
    results.append({
        "SKU": SKU,
        "Naive_MAPE": naive_mape,
        "Prophet_MAPE": prophet_mape,
        "LightGBM_MAPE": lgb_mape
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("forecast_evaluation_results.csv", index=False)
print(results_df.head()) 