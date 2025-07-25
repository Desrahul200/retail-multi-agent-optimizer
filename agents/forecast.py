from functools import lru_cache
from prophet import Prophet
import pandas as pd

@lru_cache(maxsize=64)
def _fit_prophet(product_id: str, full_df: pd.DataFrame):
    sub = full_df[full_df["product_id"] == product_id]
    prophet_df = sub[["date","units_sold"]].rename(columns={'date':'ds','units_sold':'y'})
    m = Prophet(yearly_seasonality=True, daily_seasonality=False)  # type: ignore
    m.fit(prophet_df)
    return m

def run(state: dict, full_df: pd.DataFrame) -> dict:
    pid = state["product_id"]
    sub = full_df[full_df["product_id"] == pid]
    if sub.empty:
        state["forecast"] = None
        return state
    prophet_df = sub[["date","units_sold"]].rename(columns={'date':'ds','units_sold':'y'})
    m = Prophet(yearly_seasonality=True, daily_seasonality=False)  # type: ignore
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    state["forecast"] = forecast
    return state 