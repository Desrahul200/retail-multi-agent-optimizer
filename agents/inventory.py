import pandas as pd
import numpy as np

REORDER_POINT  = 60          # trigger
REORDER_QTY    = 200         # pallet size
LEAD_TIME_ROWS = 5           # 5 rows  ≈ 5 weeks (one row = one week)

def run(state: dict, full_df: pd.DataFrame) -> dict:
    df = (
        full_df[full_df["product_id"] == state["product_id"]]
        .reset_index(drop=True)
    )
    if df.empty:
        return state

    # ------------------------------------------------------------------ #
    current      = df.loc[0, "stock_level"]
    on_order     = []                  # [(arrival_row, qty), …]
    stock_trace  = []                  # plotted line
    events_txt   = []                  # annotations (“Placed PO…”, “Received …”)

    for i, row in df.iterrows():

        # 1️⃣ receive anything that arrives *this* row
        arrivals = [q for (r, q) in on_order if r == i]
        if arrivals:
            received = sum(arrivals)
            current += received
            events_txt.append((row["date"], f"Received {received} units"))
            on_order = [(r, q) for (r, q) in on_order if r != i]

        # 2️⃣ log **pre-sale** stock so the jump is visible
        stock_trace.append(current)

        # 3️⃣ subtract demand
        current = max(current - row["units_sold"], 0)

        # 4️⃣ place PO *after* demand, if needed and none already outstanding
        if current <= REORDER_POINT and not on_order:
            arrival_row = i + LEAD_TIME_ROWS
            on_order.append((arrival_row, REORDER_QTY))
            events_txt.append((row["date"], "Placed PO for 200"))

    # ------------------------------------------------------------------ #
    state["stock_df"]   = pd.DataFrame(
        {"date": df["date"], "simulated_stock": stock_trace}
    )
    state["inv_events"] = events_txt
    return state 