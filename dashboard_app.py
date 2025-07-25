# dashboard_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from graphs.retail_graph import GRAPH as graph

# --- look-up table ----------------------------------------------------
import pandas as pd
prod_meta = pd.read_csv("product.csv")[["PRODUCT_ID", "SUB_COMMODITY_DESC"]]
name_map  = dict(zip(prod_meta["PRODUCT_ID"], prod_meta["SUB_COMMODITY_DESC"]))

st.set_page_config(layout="wide")
st.title("🛒 E-Com Multi-Agent Optimizer")

# --- Sidebar controls ---
df = pd.read_csv("dh_demand.csv")
product_list = list(df["product_id"].unique())

# --- Sidebar ----------------------------------------------------------
options = [
    f"{pid} – {name_map.get(pid, 'Unknown')}"
    for pid in product_list
]
choice = st.sidebar.selectbox("Product", options=options)

# extract the raw PRODUCT_ID back out
product = choice.split(" – ")[0]
try:
    product = int(product)
except ValueError:
    pass

# Get price range for selected product
prod_df = df[df["product_id"] == product]
min_p = float(prod_df["price"].min() * 0.8)
max_p = float(prod_df["price"].max() * 1.2)
price_override = st.sidebar.slider(
    "Manual Price",
    min_value=round(min_p, 2),
    max_value=round(max_p, 2),
    value=None, step=0.05)
promo_flag = st.sidebar.checkbox("Force Promotion", value=False)
comp_delta = st.sidebar.slider("Competitor Price Δ", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)

# Build initial state
state = {"product_id": product}
if "bandit_alpha" in st.session_state:
    state["bandit_alpha"] = st.session_state["bandit_alpha"]
    state["bandit_beta"]  = st.session_state["bandit_beta"]
if price_override is not None:
    state["manual_price_changes"] = {d: price_override for d in range(30)}  # type: ignore
if promo_flag:
    state["manual_promo_starts"] = {d: 1 for d in range(30)}  # type: ignore
else:
    state["manual_promo_starts"] = {}  # type: ignore
state["manual_competitor_drops"] = {d: comp_delta for d in range(30)}  # type: ignore

# --- Run the multi-agent graph ---
with st.spinner("Running Optimization…"):
    result = graph.invoke(state)

if "bandit_alpha" in result:
    st.session_state["bandit_alpha"] = result["bandit_alpha"]
    st.session_state["bandit_beta"]  = result["bandit_beta"]

# --- Forecast Tab ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Forecast", "Pricing", "Inventory", "Scenario", "Customers"
])

pretty_name = name_map.get(product, "")

with tab1:
    st.subheader(f"📈 30-Day Demand Forecast · {pretty_name}")

    sub = df[df["product_id"] == product]       # actual history
    if result.get("forecast") is None or sub.empty:
        st.warning("Not enough history to fit a forecast.")
    else:
        fc = result["forecast"]

        # ── split: everything after CUT-OFF is forecast ────────────────────
        cutoff = sub["date"].max()
        hist   = sub.rename(columns={"date": "ds", "units_sold": "y"})
        fut    = fc[fc["ds"] > cutoff]

        fig = go.Figure()

        # 1️⃣ ACTUAL
        fig.add_scatter(
            x=hist["ds"], y=hist["y"],
            mode="lines", line=dict(width=1.5, color="#e2e2e2"),
            name="Actual"
        )

        # 2️⃣ FORECAST
        fig.add_scatter(
            x=fut["ds"], y=fut["yhat"],
            mode="lines", line=dict(width=2.5, dash="dash", color="#4ec3ff"),
            name="Forecast"
        )

        # 3️⃣ UNCERTAINTY (shaded ribbon)
        fig.add_scatter(
            x=list(fut["ds"]) + list(fut["ds"])[::-1],
            y=list(fut["yhat_upper"]) + list(fut["yhat_lower"])[::-1],
            fill="toself", fillcolor="rgba(78,195,255,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", showlegend=False
        )

        # 4️⃣ FORECAST WINDOW BACKGROUND
        fig.add_vrect(
            x0=cutoff, x1=fut["ds"].max(),
            fillcolor="rgba(78,195,255,0.07)", layer="below", line_width=0
        )

        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=40),
            xaxis_title="Calendar date",
            yaxis_title="Units / week",
            legend=dict(orientation="h", x=0.01, y=1.12)
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"💰 Price Optimization · {pretty_name}")
    if not all(k in result and result[k] is not None for k in ["recommended_price","expected_demand","expected_revenue","revenue_curve"]):
        st.warning("Not enough data to fit a price optimization model.")
    else:
        price = result["recommended_price"]
        demand = result["expected_demand"]
        revenue = result["expected_revenue"]
        st.metric("Recommended Price", f"${price:.2f}")
        st.metric("Expected Demand", f"{demand:.1f} units")
        st.metric("Expected Revenue", f"${revenue:.2f}")

        prices = result["revenue_curve"]["price"]
        revenues = result["revenue_curve"]["revenue"]
        positive = revenues > 0
        if not positive.any():
            st.warning("Model predicted zero or negative revenue in observed price range.")
        else:
            fig2 = px.line(x=prices[positive], y=revenues[positive], labels={"x":"Price","y":"Revenue"}, title="Revenue Curve")
            fig2.update_yaxes(range=[0, revenues[positive].max() * 1.1])
            fig2.add_scatter(x=[price], y=[revenue], mode="markers",
                             marker=dict(size=10, color="red"), name="Chosen price")
            st.plotly_chart(fig2, use_container_width=True)

with tab3:
    REORDER_POINT = 60
    st.subheader(f"📦 Inventory Simulation · {pretty_name}")

    inv  = result["stock_df"]
    evts = result.get("inv_events", [])

    fig3 = go.Figure()
    fig3.add_scatter(
        x=inv["date"], y=inv["simulated_stock"],
        mode="lines",
        name="Stock level",
        line=dict(shape="hv", color="#4ec3ff")
    )

    # Reorder-point line (white, thicker)
    fig3.add_hline(
        y=REORDER_POINT, line_dash="dash",
        line_color="white", line_width=2,
        annotation_text="Reorder point", annotation_position="top right",
        annotation_font_color="white", annotation_bgcolor="#4ec3ff"
    )

    # Dots for PO events with index
    placed_x, placed_y, placed_text = [], [], []
    received_x, received_y, received_text = [], [], []
    placed_idx, received_idx = 1, 1
    for d, txt in evts:
        if "Placed PO" in txt:
            placed_x.append(d)
            placed_y.append(REORDER_POINT + 20)
            placed_text.append(str(placed_idx))
            placed_idx += 1
        elif "Received" in txt:
            received_x.append(d)
            received_y.append(REORDER_POINT - 20)
            received_text.append(str(received_idx))
            received_idx += 1

    if placed_x:
        fig3.add_scatter(
            x=placed_x, y=placed_y, mode="markers+text",
            marker=dict(size=12, color="#ffb347"),
            text=placed_text, textposition="top center",
            name="Placed PO"
        )
    if received_x:
        fig3.add_scatter(
            x=received_x, y=received_y, mode="markers+text",
            marker=dict(size=12, color="#90ee90"),
            text=received_text, textposition="bottom center",
            name="Received"
        )

    fig3.update_layout(
        yaxis_title="Units on hand",
        xaxis_title="Date",
        margin=dict(l=20, r=20, t=40, b=40)
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader(f"🧪 Scenario Simulation · {pretty_name}")
    sim_df = result["scenario_result"]
    # Show scenario metrics
    color = "red" if result["service_level"] < 95 else "white"
    st.metric("On-shelf availability", f"{result['service_level']} %")
    st.metric("Avg stock",      f"{result['avg_stock']} units")
    st.metric("POs placed",     result['orders'])
    fig4 = px.line(sim_df, x="date", y=["units_sold","stock_level"], 
                   labels={"value":"Units/Stock","variable":"Metric"},
                   title="Units Sold & Stock Level")
    st.plotly_chart(fig4, use_container_width=True)

    st.write("### Simulation Data (first 5 days)")
    st.dataframe(sim_df.head())
    st.write("### …and last 5 days")
    st.dataframe(sim_df.tail())

with tab5:
    st.subheader(f"👥 Customer Segments · {pretty_name}")

    if "segments" not in result:
        st.info("Segmentation not available for this SKU.")
    else:
        seg_df = pd.DataFrame({
            "x": result["umap_x"],
            "y": result["umap_y"],
            "segment": result["segments"]
        })
        fig = px.scatter(
            seg_df, x="x", y="y", color="segment",
            title="UMAP projection – each dot = transaction-day",
            hover_data={
                "price":   sub["price"],
                "promo":   sub["promotion_flag"],
                "comp":    sub["competitor_price"],
                "units":   sub["units_sold"],
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # quick stats table
        avg_units = (
            df[df["product_id"] == product]
            .assign(segment=result["segments"])
            .groupby("segment")["units_sold"].mean()
            .round(1)
            .rename("avg_units")
        )
        st.write("**Average units sold by segment:**")
        st.dataframe(avg_units)

