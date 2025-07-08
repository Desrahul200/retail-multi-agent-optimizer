import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('synthetic_demand.csv', parse_dates=['date'])
products = list(df['product_id'].unique())

# Simulation parameters
SIM_DAYS = 30
REORDER_POINT = 60
REORDER_QUANTITY = 200
LEAD_TIME = 5

# User scenario interventions (edit as needed)
manual_price_changes = {10: 15.0, 20: 12.0}  # day: new price
manual_promo_starts = {5: 1, 18: 1}           # day: promo flag
manual_competitor_drops = {12: -5.0}          # day: competitor price drop
new_product_launch = {25: 'Product_11'}       # day: new product_id

# Start with the last 30 days of the first product as base
target_product = products[0]
product_df = df[df['product_id'] == target_product].copy().reset_index(drop=True)
base = product_df.iloc[-1]

sim_dates = pd.date_range(base['date'] + pd.Timedelta(days=1), periods=SIM_DAYS)
sim_data = []
current_stock = base['stock_level']
lead_time_queue = []

# Fit demand model (Prophet)
prophet_df = product_df[['date', 'units_sold']].rename(columns={'date': 'ds', 'units_sold': 'y'})
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(prophet_df)

# Fit pricing model (Linear Regression)
df_model = product_df[['price', 'promotion_flag', 'competitor_price', 'units_sold']].dropna()
X = df_model[['price', 'promotion_flag', 'competitor_price']]
y = df_model['units_sold']
glm = LinearRegression().fit(X, y)

for day, sim_date in enumerate(sim_dates):
    # Manual interventions
    price = manual_price_changes.get(day, base['price'])
    promo = manual_promo_starts.get(day, 0)
    comp_price = base['competitor_price'] + manual_competitor_drops.get(day, 0)
    # Demand forecast (Prophet, fallback to regression for scenario)
    X_pred = np.array([[price, promo, comp_price]])
    demand_pred = glm.predict(X_pred)[0]
    # Inventory simulation
    arrivals = [q for (d, q) in lead_time_queue if d == day]
    if arrivals:
        current_stock += sum(arrivals)
        lead_time_queue = [(d, q) for (d, q) in lead_time_queue if d != day]
    units_sold = int(min(demand_pred, current_stock))
    current_stock -= units_sold
    current_stock = max(current_stock, 0)
    # Reorder logic
    if current_stock < REORDER_POINT and not any(d > day for (d, q) in lead_time_queue):
        lead_time_queue.append((day + LEAD_TIME, REORDER_QUANTITY))
    # New product launch
    if day in new_product_launch:
        print(f"New product launched: {new_product_launch[day]} on day {day}")
    sim_data.append({
        'date': sim_date,
        'product_id': target_product,
        'price': price,
        'promotion_flag': promo,
        'competitor_price': comp_price,
        'units_sold': units_sold,
        'stock_level': current_stock
    })

sim_df = pd.DataFrame(sim_data)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(sim_df['date'], sim_df['units_sold'], label='Units Sold')
plt.plot(sim_df['date'], sim_df['stock_level'], label='Stock Level')
plt.title(f'Scenario Simulation for {target_product}')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.show()

print("\nSimulation complete. Last 5 days:")
print(sim_df.tail()) 