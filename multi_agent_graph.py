from typing import TypedDict, Dict
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import umap
from sklearn.cluster import KMeans
from langgraph.graph import StateGraph
from functools import lru_cache

# ─── Load global dataset ───────────────────────────────────────────────────────
df = pd.read_csv("dh_demand.csv", parse_dates=["date"])
# df = pd.read_csv("synthetic_demand.csv", parse_dates=["date"])  # synthetic

PRODUCTS = df['product_id'].unique().tolist()

# ─── Define state schema ──────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    product_id: str
    segment_info: Dict[int, float]
    forecast: pd.DataFrame
    recommended_price: float
    expected_demand: float
    expected_revenue: float
    revenue_curve: pd.DataFrame
    stock_df: pd.DataFrame
    scenario_result: pd.DataFrame
    manual_price_changes: Dict[int, float]
    manual_promo_starts: Dict[int, int]
    manual_competitor_drops: Dict[int, float]

# ─── Agent 1: Customer Behavior ────────────────────────────────────────────────
def customer_behavior_agent(state: AgentState) -> AgentState:
    sub = df[df['product_id'] == state['product_id']]
    if sub.empty:
        # Optionally, return state or set a flag
        return state
    feats = sub[['price','promotion_flag','competitor_price','units_sold']].values
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
    emb = reducer.fit_transform(feats)
    kmeans = KMeans(n_clusters=3, random_state=42)
    segs = kmeans.fit_predict(emb)
    sub = sub.copy()
    sub['segment'] = segs
    state['segment_info'] = sub.groupby('segment')['units_sold'].mean().to_dict()
    return state

# ─── Agent 2: Demand Forecast ─────────────────────────────────────────────────
def demand_agent(state: AgentState) -> AgentState:
    sub = df[df['product_id'] == state['product_id']]
    if sub.empty:
        # Optionally, set a flag or message
        state['forecast'] = None
        return state
    model = fit_prophet(state['product_id'])
    prophet_df = sub[['date','units_sold']].rename(columns={'date':'ds','units_sold':'y'})
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    state['forecast'] = forecast
    return state

@lru_cache(maxsize=64)
def fit_prophet(product_id: str):
    sub = df[df["product_id"] == product_id]
    prophet_df = sub[["date","units_sold"]].rename(columns={'date':'ds','units_sold':'y'})
    m = Prophet(yearly_seasonality=True, daily_seasonality=False)
    m.fit(prophet_df)
    return m

# ─── Agent 3: Pricing Recommendation ─────────────────────────────────────────
# ─── Agent 3: Pricing Recommendation ─────────────────────────────
def pricing_agent(state: AgentState) -> AgentState:
    sub = df[df["product_id"] == state["product_id"]].copy()
    if sub.empty:
        return state                       # safety-exit

    # 1⃣  Log-log model  → always positive demand, slope = elasticity
    X = np.column_stack([
        np.log(sub["price"]),
        sub["promotion_flag"],
        sub["competitor_price"]
    ])
    y = np.log(sub["units_sold"])
    glm = LinearRegression().fit(X, y)

    last = sub.iloc[-1]
    promo, comp_price = last["promotion_flag"], last["competitor_price"]

    # 2⃣  Build a price grid around *observed* prices, not 10-50
    lo = sub["price"].min() * 0.8           # 20 % below min
    hi = sub["price"].max() * 1.2           # 20 % above max
    price_grid = np.linspace(lo, hi, 40)

    # 3⃣  Predict demand on that grid
    log_pred = glm.predict(np.column_stack([
        np.log(price_grid),
        np.full_like(price_grid, promo),
        np.full_like(price_grid, comp_price)
    ]))
    demand_pred = np.exp(log_pred)          # back to units (positive)
    revenue     = price_grid * demand_pred

    # 4⃣  Pick best price inside the realistic range
    best_idx = np.argmax(revenue)
    state["recommended_price"] = float(price_grid[best_idx])
    state["expected_demand"]   = float(demand_pred[best_idx])
    state["expected_revenue"]  = float(revenue[best_idx])
    state["revenue_curve"]     = pd.DataFrame(
        {"price": price_grid, "revenue": revenue}
    )
    return state

# ─── Agent 4: Inventory Simulation ───────────────────────────────────────────
def inventory_agent(state: AgentState) -> AgentState:
    sub = df[df['product_id'] == state['product_id']].reset_index(drop=True)
    if sub.empty:
        # Optionally, return state or set a flag
        return state
    REORDER_POINT, REORDER_Q, LEAD_TIME = 60, 200, 5

    stock = []
    queue = []  # list of (arrival_idx, qty)
    current = sub.loc[0, 'stock_level']

    for i, row in sub.iterrows():
        # receive any orders
        arrivals = [q for d, q in queue if d == i]
        if arrivals:
            current += sum(arrivals)
            queue = [(d,q) for d,q in queue if d != i]
        # subtract sales
        current = max(current - row['units_sold'], 0)
        # place reorder if needed
        if current < REORDER_POINT and not any(d > i for d,_ in queue):
            queue.append((i + LEAD_TIME, REORDER_Q))
        stock.append(current)

    state['stock_df'] = pd.DataFrame({
        'date': sub['date'],
        'simulated_stock': stock
    })
    return state

# ─── Agent 5: Scenario Simulation ─────────────────────────────────────────────
def scenario_simulator_agent(state: AgentState) -> AgentState:
    sub = df[df['product_id'] == state['product_id']].reset_index(drop=True)
    if sub.empty:
        # Optionally, return state or set a flag
        return state
    # Refit models as before…
    prophet_df = sub[['date','units_sold']].rename(columns={'date':'ds','units_sold':'y'})
    dem_model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    dem_model.fit(prophet_df)
    df_m = sub[['price','promotion_flag','competitor_price','units_sold']].dropna()
    glm = LinearRegression().fit(
        df_m[['price','promotion_flag','competitor_price']],
        df_m['units_sold']
    )

    SIM_DAYS = 30
    current_stock = sub.loc[len(sub)-1, 'stock_level']
    REORDER_POINT, REORDER_Q, LEAD_TIME = 60, 200, 5
    queue = []
    base = sub.iloc[-1]
    sim = []

    # grab your manual‐override dicts (or default to empty)
    manual_prices = state.get('manual_price_changes', {})
    manual_promos = state.get('manual_promo_starts', {})
    manual_comps  = state.get('manual_competitor_drops', {})

    for day in range(SIM_DAYS):
        # price = manual override if present for that day, else use agent recommendation
        price = manual_prices.get(day, state.get('recommended_price', base['price']))
        promo = manual_promos.get(day, 0)
        comp  = base['competitor_price'] + manual_comps.get(day, 0)

        # inventory arrival
        arrivals = [q for d,q in queue if d == day]
        if arrivals:
            current_stock += sum(arrivals)
            queue = [(d,q) for d,q in queue if d != day]

        stock_start = current_stock  # ⚡ record starting stock **before** sales

        # demand via regression
        demand_pred = max(0, glm.predict([[price, promo, comp]])[0])

        # sales & reorder logic
        sold = int(min(demand_pred, current_stock))
        current_stock = max(current_stock - sold, 0)
        if current_stock < REORDER_POINT and not any(d > day for (d,_) in queue):
            queue.append((day + LEAD_TIME, REORDER_Q))

        sim.append({
            'date': base['date'] + pd.Timedelta(days=day+1),
            'price': price,
            'promotion_flag': promo,
            'competitor_price': comp,
            'stock_start': stock_start,  # ⚡ new column
            'units_sold': sold,
            'stock_level': current_stock
        })

    state['scenario_result'] = pd.DataFrame(sim)
    return state


# ─── Build & Compile the LangGraph ───────────────────────────────────────────
builder = StateGraph(AgentState)
builder.add_node("CustomerBehaviorAgent", customer_behavior_agent)
builder.add_node("DemandAgent",           demand_agent)
builder.add_node("PricingAgent",          pricing_agent)
builder.add_node("InventoryAgent",        inventory_agent)
builder.add_node("ScenarioSimulatorAgent",scenario_simulator_agent)

builder.set_entry_point("CustomerBehaviorAgent")
builder.add_edge("CustomerBehaviorAgent", "DemandAgent")
builder.add_edge("DemandAgent",           "PricingAgent")
builder.add_edge("PricingAgent",          "InventoryAgent")
builder.add_edge("InventoryAgent",        "ScenarioSimulatorAgent")
builder.set_finish_point("ScenarioSimulatorAgent")

graph = builder.compile()

if __name__ == "__main__":
    initial = {"product_id": PRODUCTS[0]}
    result = graph.invoke(initial)
    print("\n--- Final Scenario Output ---")
    print(result['scenario_result'].tail())
