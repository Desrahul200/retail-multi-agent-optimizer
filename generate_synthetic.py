# generate_synthetic.py
import numpy as np, pandas as pd, random
from datetime import date, timedelta

# ───────── parameters ──────────────────────────────────────────
DAYS, NUM_PROD, NUM_CAT = 365, 10, 3
START_DATE = date(2023, 1, 1)
OUTFILE = "synthetic_demand.csv"

# products & categories
cats     = [f"Category_{i+1}" for i in range(NUM_CAT)]
products = [f"Product_{i+1}"  for i in range(NUM_PROD)]
prod_cat = {p: random.choice(cats) for p in products}

# per-product latent variables
season_fac = {p: np.random.uniform(0.7, 1.3)  for p in products}
demand_base= {p: np.random.uniform(20, 100)   for p in products}
elasticity = {p: np.random.uniform(-2, -0.5)  for p in products}

PROMO_BOOST, COMP_EFF = 1.30, -0.70
STOCK_MIN,  STOCK_MAX = 50,  300

rows = []
for d in range(DAYS):
    day_date = START_DATE + timedelta(days=d)
    sin_mult = 1 + 0.20*np.sin(2*np.pi*d/365)     # yearly seasonality
    for p in products:
        price      = round(np.random.uniform(10, 50), 2)
        comp_price = round(price + np.random.uniform(-5, 5), 2)
        promo_flag = np.random.binomial(1, 0.15)
        stock      = np.random.randint(STOCK_MIN, STOCK_MAX)

        demand = demand_base[p]*season_fac[p]*sin_mult
        demand *= (price/30)**elasticity[p]
        if promo_flag:          demand *= PROMO_BOOST
        demand *= 1 + COMP_EFF*(price-comp_price)/price
        demand += np.random.normal(0, 3)

        sold = int(max(0, min(stock, round(demand))))
        rows.append(dict(date=day_date,
                         product_id=p,
                         category=prod_cat[p],
                         price=price,
                         units_sold=sold,
                         promotion_flag=promo_flag,
                         competitor_price=comp_price,
                         stock_level=stock))

pd.DataFrame(rows).to_csv(OUTFILE, index=False)
print(f"✅   synthetic dataset → {OUTFILE}") 