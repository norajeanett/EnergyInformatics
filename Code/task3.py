import datetime
import requests
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

###############################################################################
# 1) BASE APPLIANCE DICTIONARY (SINGLE HOUSEHOLD)
#    We'll replicate these for multiple households, but only some fraction
#    (now 7 of them) will include an EV.
###############################################################################

base_appliances = {
    # MUST-RUN APPLIANCES (LIGHTING, HEATING, FRIDGE, TV, WI-FI)
    "Lighting": {
        "E": 2.0,
        "intervals": [(11, 20)],  # hours 11..20
        "gamma_max": 0.3,
        "must_run": True,
        "color": "gold"
    },
    "Heating": {
        "E": 9.6,
        "intervals": [(1, 24)],
        "gamma_max": 2.0,
        "must_run": True,
        "color": "red"
    },
    "Refrigerator": {
        "E": 1.32,
        "intervals": [(1, 24)],
        "gamma_max": 0.165,
        "must_run": True,
        "color": "green"
    },
    "TV": {
        "E": 0.6,
        "intervals": [(6, 8), (20, 23)],
        "gamma_max": 0.12,
        "must_run": True,
        "color": "purple"
    },
    "Wi-Fi": {
        "E": 0.006,
        "intervals": [(1, 24)],
        "gamma_max": 0.010,
        "must_run": True,
        "color": "lime"
    },

    # SHIFTABLE (Stove, Computer, Laundry, etc.)
    "ElectricStove": {
        "E": 3.9,
        "intervals": [(16, 18)],
        "gamma_max": 5.0,
        "must_run": False,
        "color": "brown"
    },
    "Computer": {
        "E": 0.6,
        "intervals": [(6, 8), (16, 23)],
        "gamma_max": 0.14,
        "must_run": False,
        "color": "darkblue"
    },
    "Laundry Machine": {
        "E": 1.94,
        "intervals": [(6, 8), (16, 23)],
        "gamma_max": 2.0,
        "must_run": False,
        "color": "pink"
    },
    "Dishwasher": {
        "E": 1.44,
        "intervals": [(6, 8), (16, 23)],
        "gamma_max": 1.5,
        "must_run": False,
        "color": "cyan"
    },
    "Coffee Maker": {
        "E": 0.9,
        "intervals": [(6, 8), (16, 23)],
        "gamma_max": 1.2,
        "must_run": False,
        "color": "orange"
    },
    "Phone Charger": {
        "E": 0.004,
        "intervals": [(6, 8), (16, 23)],
        "gamma_max": 0.010,
        "must_run": False,
        "color": "olive"
    }
}

# EV definition (to be added only if a household has EV)
ev_appliance = {
    "EV": {
        "E": 9.90,
        "intervals": [(6, 8), (16, 23)],
        "gamma_max": 7.2,
        "must_run": False,
        "color": "gray"
    }
}

###############################################################################
# 2) REPLICATE FOR N=30 HOUSEHOLDS, EXACTLY 7 OWNS EV
###############################################################################
def build_neighborhood_appliances(num_households=30, ev_count=7):
    """
    Create appliance data for a specified number of households, with a fraction owning electric vehicles (EV).

    For each household, replicates the base appliances and assigns an EV to a specified number of households.
    Constructs a dictionary where keys represent "House{i}_ApplianceName".

    Parameters:
    - num_households (int): The total number of households in the neighborhood. Defaults to 30.
    - ev_count (int): The number of households with an EV. Defaults to 7.

    Returns:
    - dict: A dictionary mapping unique appliance names to their properties, tailored to each household.
    """
    all_appliances = {}
    # We'll let the first `ev_count` households own an EV.
    ev_house_ids = set(range(ev_count))  # e.g. 0..6 => 7 houses total

    for i in range(num_households):
        house_label = f"House{i+1}"  # i+1 => House1..House30
        # Copy the base
        for (ap_name, ap_info) in base_appliances.items():
            unique_name = f"{house_label}_{ap_name}"
            all_appliances[unique_name] = dict(ap_info)  # shallow copy

        # If this house has EV:
        if i in ev_house_ids:
            unique_ev_name = f"{house_label}_EV"
            all_appliances[unique_ev_name] = dict(ev_appliance["EV"])

    return all_appliances

###############################################################################
# 3) Convert dictionary to pmin/pmax
###############################################################################
def forced_usage(e_total, hour_count):
    """
    Calculate the energy usage rate per hour for a must-run appliance.

    Parameters:
    - e_total (float): Total energy consumption required for the appliance over its run period (kWh).
    - hour_count (int): The number of hours the appliance is allowed to run.

    Returns:
    - float: The average energy consumption per hour (kWh/hour). Returns 0.0 if hour_count is less than 1.
    """
    if hour_count < 1:
        return 0.0
    return e_total / hour_count

def build_pmin_pmax_dict(appliances_raw):
    """
    Prepare minimum and maximum power usage parameters for each appliance.

    For must-run appliances, calculates strict hourly energy rates. For shiftable appliances, sets bounds based on maximum power allowed.

    Parameters:
    - appliances_raw (dict): Dictionary of appliance information including energy requirements and operational intervals.

    Returns:
    - dict: A dictionary with updated appliance information including 'pmin' and 'pmax' for scheduling.
    """
    updated = {}
    for a_name, info in appliances_raw.items():
        E_a = info["E"]
        intervals = info["intervals"]
        gamma_max = info["gamma_max"]
        must_run_flag = info.get("must_run", False)
        color = info["color"]

        # Count hours in intervals
        hour_count = 0
        for (start, end) in intervals:
            hour_count += (end - start + 1)

        if must_run_flag:
            # forced usage each hour
            rate = forced_usage(E_a, hour_count)
            pmin_val = rate
            pmax_val = rate
        else:
            pmin_val = 0.0
            pmax_val = gamma_max

        updated[a_name] = {
            "E": E_a,
            "intervals": intervals,
            "pmin": pmin_val,
            "pmax": pmax_val,
            "color": color
        }
    return updated


def fetch_no1_prices_for_today():
    """
    Fetch today's electricity prices for the NO1 region from 'Hva koster strømmen' service.

    Returns:
    - dict: A dictionary with hour keys (1-24) and the corresponding electricity price in NOK/kWh.

    Raises:
    - requests.exceptions.HTTPError: If the data fetching fails, an exception is raised.
    """
    today = datetime.date.today()
    yyyy = today.strftime("%Y")
    mmdd = today.strftime("%m-%d")
    url = f"https://www.hvakosterstrommen.no/api/v1/prices/{yyyy}/{mmdd}_NO1.json"

    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    prices = {}
    for i, entry in enumerate(data):
        hour_idx = i+1
        prices[hour_idx] = entry["NOK_per_kWh"]
    return prices


def solve_scheduling_lp(prices, appliances_dict):
    """
    Solve a linear programming problem to minimize the daily cost of appliance energy consumption.

    Objective is to minimize electricity costs based on hourly prices, subject to constraints on appliances' energy use and operational intervals.

    Parameters:
    - prices (dict): Hourly electricity prices in NOK/kWh.
    - appliances_dict (dict): Appliance data including energy requirements and operational constraints.

    Returns:
    - tuple: A dictionary of appliance usage by hour (x_dict) and the minimized total cost (float).

    Raises:
    - prints error message if linear programming solver fails.
    """
    A_all = list(appliances_dict.keys())
    nA = len(A_all)
    nH = 24
    total_vars = nA * nH

    def var_idx(a_idx, hour):
        return a_idx * nH + (hour - 1)

    # Objective
    c = np.zeros(total_vars)
    for a_idx, a_name in enumerate(A_all):
        for h in range(1, nH+1):
            c[var_idx(a_idx, h)] = prices[h]

    # sum_{h} x_{a,h} = E_a
    A_eq = np.zeros((nA, total_vars))
    b_eq = np.zeros(nA)
    for a_idx, a_name in enumerate(A_all):
        E_a = appliances_dict[a_name]["E"]
        for h in range(1, nH+1):
            A_eq[a_idx, var_idx(a_idx, h)] = 1.0
        b_eq[a_idx] = E_a

    # Bounds
    bounds = []
    for a_idx, a_name in enumerate(A_all):
        info = appliances_dict[a_name]
        pmin_a = info["pmin"]
        pmax_a = info["pmax"]
        intervals = info["intervals"]

        valid_hours = set()
        for (start, end) in intervals:
            for hh in range(start, end+1):
                valid_hours.add(hh)

        for h in range(1, nH+1):
            if h in valid_hours:
                bounds.append((pmin_a, pmax_a))
            else:
                # not in intervals => x=0
                bounds.append((0.0, 0.0))

    res = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs'
    )
    if not res.success:
        print("linprog failed:", res.message)
        return None, None

    x_solution = res.x
    # parse solution
    x_dict = {}
    for a_idx, a_name in enumerate(A_all):
        x_dict[a_name] = {}
        for h in range(1, nH+1):
            x_dict[a_name][h] = x_solution[var_idx(a_idx, h)]

    total_cost = res.fun
    return x_dict, total_cost

def plot_price_curve(prices):
    """
    Plot the hourly electricity price curve for the NO1 region.

    Parameters:
    - prices (dict): Hourly electricity prices in NOK/kWh.

    Returns:
    None. Displays the price curve plot.
    """
    hours = range(1,25)
    vals = [prices[h] for h in hours]
    plt.figure(figsize=(8,4))
    plt.plot(hours, vals, marker='o')
    plt.title("Hourly Electricity Price (NO1)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Price (NOK/kWh)")
    plt.xticks(hours)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_neighborhood_usage(x_dict, appliances_dict, prices):
    """
    Visualize total energy usage by hour for all households in the neighborhood.

    Parameters:
    - x_dict (dict): Hourly energy usage data for each appliance.
    - appliances_dict (dict): Appliance data for visualization context.
    - prices (dict): Hourly electricity prices used for context.

    Returns:
    None. Displays usage bar chart for the entire neighborhood.
    """
    hours = list(range(1,25))
    usage_by_hour = []
    for h in hours:
        usage_h = sum(x_dict[a][h] for a in x_dict)
        usage_by_hour.append(usage_h)

    plt.figure(figsize=(10,5))
    plt.bar(hours, usage_by_hour, edgecolor="black")
    plt.title("Total Neighborhood Usage by Hour (All Households)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Usage (kWh)")
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_hourly_cost_neighborhood(x_dict, appliances_dict, prices):
    """
    Plot the neighborhood's total hourly cost of electricity based on scheduled appliance usage.

    Parameters:
    - x_dict (dict): Hourly energy usage for each appliance.
    - appliances_dict (dict): Appliance data for context.
    - prices (dict): Hourly electricity prices in NOK/kWh.

    Returns:
    None. Displays bar chart of hourly costs and prints the total daily cost.
    """
    hours = range(1,25)
    cost_hourly = []
    for h in hours:
        usage_sum = sum(x_dict[a][h] for a in x_dict)
        cost_hourly.append(usage_sum * prices[h])
    total_cost = sum(cost_hourly)

    plt.figure(figsize=(10,4))
    plt.bar(hours, cost_hourly, color="skyblue", edgecolor="black")
    plt.title(f"Neighborhood Hourly Cost (Total={total_cost:.2f} NOK)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Cost (NOK)")
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print(f"Total daily cost (recomputed from hours): {total_cost:.2f} NOK")


def main():
    """
    Executes the main workflow for optimizing appliance scheduling in a neighborhood.

    Tasks:
    - Builds data for a number of households, some with EVs.
    - Converts appliance data into scheduling constraints.
    - Attempts to fetch electricity prices, applying fallback if necessary.
    - Solves the linear programming optimization problem.
    - Summarizes and prints household energy usage.
    - Generates plots to visualize prices, usage, and costs.

    Returns:
    None. Runs through the sequence of operations and displays results.
    """
    # Build big dictionary for 30 households, exactly 7 have EV
    all_houses_dict = build_neighborhood_appliances(num_households=30, ev_count=7)

    # Convert to pmin/pmax
    final_appliances = build_pmin_pmax_dict(all_houses_dict)

    # Fetch real-time prices or fallback
    try:
        prices = fetch_no1_prices_for_today()
    except Exception as e:
        print("Warning: fallback due to:", e)
        prices = {h:1.0 for h in range(1,25)}

    # Solve the large LP
    x_dict, total_cost = solve_scheduling_lp(prices, final_appliances)
    if x_dict is None:
        print("No feasible solution, aborting.")
        return

    print("\nlinprog: Optimal solution found for the Neighborhood")
    print(f"Optimal total daily cost: {total_cost:.2f} NOK\n")

    # Summarize usage per house
    house_usage = {}
    for a_name in x_dict:
        # a_name looks like "HouseX_Appliance"
        house_label = a_name.split("_")[0]
        if house_label not in house_usage:
            house_usage[house_label] = 0.0
        usage_a = sum(x_dict[a_name][h] for h in range(1,25))
        house_usage[house_label] += usage_a

    for h_label in sorted(house_usage.keys()):
        print(f"{h_label} total usage = {house_usage[h_label]:.3f} kWh")

    # Plots
    plot_price_curve(prices)
    plot_neighborhood_usage(x_dict, final_appliances, prices)
    plot_hourly_cost_neighborhood(x_dict, final_appliances, prices)

if __name__ == "__main__":
    main()
