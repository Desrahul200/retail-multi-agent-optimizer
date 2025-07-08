import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
REORDER_POINT = 60  # Alert if stock falls below this
REORDER_QUANTITY = 200  # How much to order when reordering
LEAD_TIME = 5  # Days between order and restock

# Load data
df = pd.read_csv('synthetic_demand.csv', parse_dates=['date'])
product_id = df['product_id'].unique()[0]
product_df = df[df['product_id'] == product_id].copy().reset_index(drop=True)

# Simulate inventory
stock = []
orders = []
alerts = []
current = product_df.iloc[-1]['stock_level']
lead_time_queue = []  # (arrival_day, quantity)

for i, row in product_df.iterrows():
    # Receive any arriving orders
    arrivals = [q for (d, q) in lead_time_queue if d == i]
    if arrivals:
        current += sum(arrivals)
        lead_time_queue = [(d, q) for (d, q) in lead_time_queue if d != i]
    # Subtract sales
    current -= row['units_sold']
    current = max(current, 0)
    # Check for reorder
    if current < REORDER_POINT and not any(d > i for (d, q) in lead_time_queue):
        lead_time_queue.append((i + LEAD_TIME, REORDER_QUANTITY))
        orders.append(i)
        alerts.append((row['date'], 'Reorder placed'))
    # Alert for low inventory or stockout
    if current < REORDER_POINT:
        alerts.append((row['date'], 'Low inventory'))
    if current == 0:
        alerts.append((row['date'], 'Stockout'))
    stock.append(current)

product_df['simulated_stock'] = stock

# Plot inventory over time
plt.figure(figsize=(10, 4))
plt.plot(product_df['date'], product_df['simulated_stock'], label='Simulated Stock Level')
plt.axhline(REORDER_POINT, color='orange', linestyle='--', label='Reorder Point')
plt.title(f'Inventory Simulation for {product_id}')
plt.xlabel('Date')
plt.ylabel('Stock Level')
plt.legend()
plt.tight_layout()
plt.show()

# Print alerts
print(f"\nAlerts for {product_id}:")
for date, msg in alerts:
    print(f"{date.date()}: {msg}") 