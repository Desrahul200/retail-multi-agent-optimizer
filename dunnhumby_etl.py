# dunnhumby_etl.py
import pandas as pd, numpy as np
from datetime import date, timedelta

RAW_TRX  = "transaction_data.csv"
RAW_PROD = "product.csv"
OUTFILE  = "dh_demand.csv"

# ─── load & clean ──────────────────────────────────────────────
trx  = pd.read_csv(RAW_TRX)
prod = pd.read_csv(RAW_PROD)

trx = trx[trx["QUANTITY"] > 0]           # drop zero-qty rows
trx = trx[trx["SALES_VALUE"] >= 0]       # drop refunds

day0 = date(2017, 1, 1)
trx["date"] = pd.to_datetime(
    trx["DAY"].apply(lambda d: day0 + timedelta(days=int(d)-1))
)

trx["promo"] = ((trx["COUPON_DISC"] != 0) |
                (trx["COUPON_MATCH_DISC"] != 0) |
                (trx["RETAIL_DISC"] != 0)).astype(int)

# ─── weekly × product aggregation ─────────────────────────────
weekly = (trx
          .groupby([pd.Grouper(key="date", freq="W"), "PRODUCT_ID"],
                   as_index=False)
          .agg(units_sold=("QUANTITY", "sum"),
               sales_value=("SALES_VALUE", "sum"),
               promotion_flag=("promo", "max"))
)
weekly["price"] = weekly["sales_value"] / weekly["units_sold"]

# keep SKUs that actually vary in price
var = weekly.groupby("PRODUCT_ID")["price"].std()
candidates = var[var > 0.25].index          # >25¢ std-dev
weekly = weekly[weekly["PRODUCT_ID"].isin(candidates)]

# scale volume so inventory simulation has something to chew
weekly["units_sold"] *= 25                 # packs of 25

# set lower starting stock
weekly["stock_level"] = 150

# attach category
weekly = weekly.merge(prod[["PRODUCT_ID","SUB_COMMODITY_DESC"]],
                      on="PRODUCT_ID", how="left")
weekly = weekly.rename(columns={"PRODUCT_ID":"product_id",
                                "SUB_COMMODITY_DESC":"category"})

# fabricate competitor price & stock
np.random.seed(42)
weekly["competitor_price"] = weekly["price"] * np.random.uniform(0.9, 1.1,
                                                                 len(weekly))

weekly = weekly[["date","product_id","category","price",
                 "units_sold","promotion_flag","competitor_price",
                 "stock_level"]]

weekly.to_csv(OUTFILE, index=False)
print(f"✅   dunnhumby weekly dataset → {OUTFILE}") 