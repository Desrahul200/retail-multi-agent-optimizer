import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('synthetic_demand.csv', parse_dates=['date'])

# Select a product to analyze
product_id = df['product_id'].unique()[0]
product_df = df[df['product_id'] == product_id].copy()

# Features and target
df_model = product_df[['price', 'promotion_flag', 'competitor_price', 'units_sold']].dropna()
X = df_model[['price', 'promotion_flag', 'competitor_price']]
y = df_model['units_sold']

# Fit linear regression
glm = LinearRegression()
glm.fit(X, y)

print(f"Price elasticity (coefficient): {glm.coef_[0]:.3f}")
print(f"Promotion effect (coefficient): {glm.coef_[1]:.3f}")
print(f"Competitor price effect (coefficient): {glm.coef_[2]:.3f}")

# Simulate price recommendations for a given day
current_row = product_df.iloc[-1]
promo = current_row['promotion_flag']
comp_price = current_row['competitor_price']

prices = np.linspace(10, 50, 40)
predicted_demand = glm.predict(np.column_stack([prices, np.full_like(prices, promo), np.full_like(prices, comp_price)]))
revenue = prices * predicted_demand

best_idx = np.argmax(revenue)
recommended_price = prices[best_idx]
expected_demand = predicted_demand[best_idx]
expected_revenue = revenue[best_idx]

print(f"\nRecommended price for {product_id}: ${recommended_price:.2f}")
print(f"Expected demand: {expected_demand:.1f} units")
print(f"Expected revenue: ${expected_revenue:.2f}")

# Plot revenue curve
plt.plot(prices, revenue, label='Expected Revenue')
plt.xlabel('Price')
plt.ylabel('Revenue')
plt.title(f'Optimal Price Recommendation for {product_id}')
plt.axvline(recommended_price, color='red', linestyle='--', label='Recommended Price')
plt.legend()
plt.tight_layout()
plt.show() 