# dashboard_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from multi_agent_graph import graph  # your LangGraph compiled graph

st.set_page_config(layout="wide")
st.title("🛒 E-Com Multi-Agent Optimizer")

# --- Sidebar controls ---
df = pd.read_csv("dh_demand.csv")
product_list = list(df["product_id"].unique())
product = st.sidebar.selectbox("Product ID", options=product_list)

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
if price_override is not None:
    # apply this same manual price every day of the 30-day sim
    state["manual_price_changes"] = {d: price_override for d in range(30)}
if promo_flag:
    state["manual_promo_starts"] = {d: 1 for d in range(30)}
else:
    state["manual_promo_starts"] = {}
state["manual_competitor_drops"] = {d: comp_delta for d in range(30)}

# --- Run the multi-agent graph ---
with st.spinner("Running Optimization…"):
    result = graph.invoke(state)

# --- Forecast Tab ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Forecast", "Pricing", "Inventory", "Scenario", "Customers"
])

with tab1:
    st.subheader("📈 30-Day Demand Forecast")
    if result.get("forecast") is None:
        st.warning("Not enough history to fit a forecast.")
    else:
        forecast_df = result["forecast"]
        plot_df = forecast_df.tail(90)  # last 60 history + 30 forecast
        fig = go.Figure()
        x = list(plot_df["ds"]) + list(plot_df["ds"])[::-1]
        y = list(plot_df["yhat_upper"]) + list(plot_df["yhat_lower"])[::-1]
        fig.add_scatter(x=plot_df["ds"], y=plot_df["yhat"], mode="lines", name="Forecast")
        fig.add_scatter(
            x=x,
            y=y,
            fill="toself", fillcolor="rgba(0,120,200,0.2)",
            line=dict(color="rgba(255,255,255,0)"), showlegend=False,
            name="Uncertainty"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("💰 Price Optimization")
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
            fig2.add_vline(x=price, line_dash="dash")
            st.plotly_chart(fig2, use_container_width=True)

with tab3:
    REORDER_POINT = 60
    st.subheader("📦 Inventory Levels")
    inv_df = result["stock_df"]
    fig3 = px.line(inv_df, x="date", y="simulated_stock", title="Inventory Over Time")
    fig3.add_hline(y=REORDER_POINT, line_dash="dash", annotation_text="Reorder Point")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("🔮 Scenario Optimizer")
    sim_df = result["scenario_result"]
    fig4 = px.line(sim_df, x="date", y=["units_sold","stock_level"], 
                   labels={"value":"Units/Stock","variable":"Metric"},
                   title="Units Sold & Stock Level")
    st.plotly_chart(fig4, use_container_width=True)

    st.write("### Simulation Data (first 5 days)")
    st.dataframe(sim_df.head())
    st.write("### …and last 5 days")
    st.dataframe(sim_df.tail())

with tab5:
    st.subheader("🧑‍💼 Customer Segments (UMAP)")

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
            title="UMAP projection – each dot = transaction-day"
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

