import pandas as pd
from prophet import Prophet
import lightgbm as lgb
import numpy as np

def run(state: dict, full_df: pd.DataFrame) -> dict:
    sub = full_df[full_df['product_id'] == state['product_id']].reset_index(drop=True)
    if sub.empty:
        return state
    prophet_df = sub[['date','units_sold']].rename(columns={'date':'ds','units_sold':'y'})
    dem_model = Prophet(yearly_seasonality=True, daily_seasonality=False)  # type: ignore
    dem_model.fit(prophet_df)
    df_m = sub[['price','promotion_flag','competitor_price','units_sold']].dropna()
    X = df_m[['price', 'promotion_flag', 'competitor_price']]
    y = df_m['units_sold']
    # LightGBM Poisson regression with monotonicity constraint: -1 for price
    model = lgb.LGBMRegressor(objective='poisson', monotone_constraints=[-1, 0, 0])
    model.fit(X, y)
    SIM_DAYS = 30
    current_stock = sub.loc[len(sub)-1, 'stock_level']
    REORDER_POINT, REORDER_Q, LEAD_TIME = 60, 200, 5
    queue = []
    base = sub.iloc[-1]
    sim = []
    manual_prices = state.get('manual_price_changes', {})
    manual_promos = state.get('manual_promo_starts', {})
    manual_comps  = state.get('manual_competitor_drops', {})
    for day in range(SIM_DAYS):
        price = manual_prices.get(day, state.get('recommended_price', base['price']))
        promo = manual_promos.get(day, 0)
        comp  = base['competitor_price'] + manual_comps.get(day, 0)
        arrivals = [q for d,q in queue if d == day]
        if arrivals:
            current_stock += sum(arrivals)
            queue = [(d,q) for d,q in queue if d != day]
        stock_start = current_stock
        X_pred = pd.DataFrame({
            "price":            [price],
            "promotion_flag":   [promo],
            "competitor_price": [comp]
        })
        demand_pred = np.asarray(model.predict(X_pred))[0]
        sold = int(min(demand_pred, current_stock))
        current_stock = max(current_stock - sold, 0)
        if current_stock < REORDER_POINT and not any(d > day for (d,_) in queue):
            queue.append((day + LEAD_TIME, REORDER_Q))
        sim.append({
            'date': base['date'] + pd.Timedelta(days=day+1),
            'price': price,
            'promotion_flag': promo,
            'competitor_price': comp,
            'stock_start': stock_start,
            'units_sold': sold,
            'stock_level': current_stock
        })
    sim_df = pd.DataFrame(sim)
    service_level = 1 - (sim_df.units_sold.eq(sim_df.stock_start) & sim_df.stock_start.eq(0)).sum() / sim_df.units_sold.sum()
    avg_on_hand   = sim_df.stock_start.mean()
    orders_placed = sim_df.price.eq(base["price"]).where(sim_df.stock_start < REORDER_POINT).count()
    state['scenario_result'] = sim_df
    state["service_level"] = round(service_level * 100, 1)   # %
    state["avg_stock"]     = int(avg_on_hand)
    state["orders"]        = int(orders_placed)
    return state 