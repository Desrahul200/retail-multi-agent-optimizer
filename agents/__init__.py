# agents/__init__.py
from .customer  import run as customer_run
from .forecast  import run as forecast_run
from .pricing   import run as pricing_run
from .inventory import run as inventory_run
from .scenario  import run as scenario_run

def get_all_agents():
    return {
        'customer': customer_run,
        'forecast': forecast_run,
        'pricing': pricing_run,
        'inventory': inventory_run,
        'scenario': scenario_run,
    } 