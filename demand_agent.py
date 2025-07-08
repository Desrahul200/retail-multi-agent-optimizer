import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('synthetic_demand.csv', parse_dates=['date'])

# Select a product to forecast
target_product = df['product_id'].unique()[0]
product_df = df[df['product_id'] == target_product].copy()

# Prepare data for Prophet
prophet_df = product_df[['date', 'units_sold']].rename(columns={'date': 'ds', 'units_sold': 'y'})

# Initialize and fit Prophet model
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(prophet_df)

# Make future dataframe (30 days ahead)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
plt.title(f'Demand Forecast for {target_product}')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.tight_layout()
plt.show()

# Print summary
print(f"Forecast for product: {target_product}")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)) 