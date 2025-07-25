from langgraph.graph import StateGraph
from agents import customer, forecast, pricing, inventory, scenario
import pandas as pd

FULL_DF = pd.read_csv("dh_demand.csv", parse_dates=["date"])

def make_graph():
    builder = StateGraph(dict)  # Generic state = Dict[str, Any]
    builder.add_node("Customer",  lambda s: customer.run(s, FULL_DF))
    builder.add_node("Forecast",  lambda s: forecast.run(s, FULL_DF))
    builder.add_node("Pricing",   lambda s: pricing.run(s, FULL_DF))
    builder.add_node("Inventory", lambda s: inventory.run(s, FULL_DF))
    builder.add_node("Scenario",  lambda s: scenario.run(s, FULL_DF))
    builder.set_entry_point("Customer")
    builder.add_edge("Customer",  "Forecast")
    builder.add_edge("Forecast",  "Pricing")
    builder.add_edge("Pricing",   "Inventory")
    builder.add_edge("Inventory", "Scenario")
    builder.set_finish_point("Scenario")
    return builder.compile()

GRAPH = make_graph() 