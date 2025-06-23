import datetime
import requests
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from appliances_2_4 import appliances_raw

###############################################################################
# APPLIANCE DICTIONARY:
#   - E: daily kWh
#   - intervals: list of (startHour, endHour) in [1..24]
#   - gamma_max: maximum kW (power rating) for shiftable usage
#   - must_run: if True => forced usage each hour = E / (# hours)
###############################################################################


def forced_usage(e_total, hour_count):
    """
    Calculate the fixed hourly energy usage for a must-run appliance.

    Parameters:
    - e_total (float): Total daily energy consumption required (kWh).
    - hour_count (int): Total hours the appliance is allowed to run per day.

    Returns:
    - float: Average energy consumption per hour. Returns 0.0 if hour_count is less than 1.
    """
    if hour_count < 1:
        return 0.0
    return e_total / hour_count

def build_pmin_pmax_dict(appliances_raw):
    """
    Construct minimum and maximum power constraints for each appliance.

    For must-run appliances, this sets strict hourly energy rates. For shiftable appliances,
    it sets bounds based on maximum power availability.

    Parameters:
    - appliances_raw (dict): Raw appliance information detailing energy needs and intervals.

    Returns:
    - dict: An updated dictionary containing pmin and pmax values for appliance scheduling.
    """
    updated = {}
    for a_name, info in appliances_raw.items():
        E_a = info["E"]
        intervals = info["intervals"]
        gamma_max = info["gamma_max"]
        must_run_flag = info["must_run"]
        color = info["color"]

        hour_count = 0
        for (start, end) in intervals:
            hour_count += (end - start + 1)

        if must_run_flag:
            # forced usage for each hour
            rate = forced_usage(E_a, hour_count)
            pmin_val = rate
            pmax_val = rate
        else:
            # shiftable => pmin=0, pmax=gamma_max
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
    Retrieve current day's electricity prices for the NO1 region from 'Hva koster strømmen' API.

    Returns:
    - dict: A dictionary mapping each hour of the day to its corresponding electricity price in NOK/kWh.

    Raises:
    - requests.exceptions.HTTPError: If there is an issue fetching the data, the exception is raised.
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
        hour_idx = i + 1
        prices[hour_idx] = entry["NOK_per_kWh"]
    return prices


def solve_scheduling_lp_with_peak(prices, appliances_dict, alpha=1.0):
    """
    Solve a linear programming problem to minimize energy and peak load costs.

    Considers not only energy usage costs but also introduces a peak load variable, 
    minimizing a combined objective including peak load penalties.

    Parameters:
    - prices (dict): Hourly electricity prices in NOK/kWh.
    - appliances_dict (dict): Appliance constraints including energy requirements and intervals.
    - alpha (float): Weighting factor for peak load in the optimization objective.

    Returns:
    - tuple: Contains three items:
        - dict: Scheduled appliance usage mapped by hour.
        - float: Peak load value (L) derived from the optimization.
        - float: Total objective value encompassing energy and peak load costs.

    Raises:
    - prints error message if solving the linear program fails.
    """
    A_all = list(appliances_dict.keys())
    nA = len(A_all)
    nH = 24

    # We'll store usage variables first, then L as the last variable:
    # usage variable indices: 0..(nA*nH - 1)
    # L index: nA*nH
    total_vars = nA*nH + 1
    L_index = nA*nH

    def var_idx(a_idx, hour):
        return a_idx*nH + (hour - 1)

    # Build objective
    c = np.zeros(total_vars)
    # cost for x_{a,h}
    for a_idx, a_name in enumerate(A_all):
        for h in range(1, nH+1):
            c[var_idx(a_idx, h)] = prices[h]
    # cost for L => alpha
    c[L_index] = alpha

    # 1) sum_{h} x_{a,h} = E_a
    A_eq = np.zeros((nA, total_vars))
    b_eq = np.zeros(nA)
    for a_idx, a_name in enumerate(A_all):
        E_a = appliances_dict[a_name]["E"]
        for h in range(1, nH+1):
            A_eq[a_idx, var_idx(a_idx,h)] = 1.0
        b_eq[a_idx] = E_a

    # 2) sum_{a} x_{a,h} <= L for each h => sum_{a} x_{a,h} - L <= 0
    # We'll have 24 such constraints, one for each hour
    A_ub = np.zeros((nH, total_vars))
    b_ub = np.zeros(nH)
    for h in range(1, nH+1):
        row_idx = h - 1
        # sum_{a} x_{a,h}
        for a_idx in range(nA):
            A_ub[row_idx, var_idx(a_idx,h)] = 1.0
        # - L
        A_ub[row_idx, L_index] = -1.0
        # <= 0 => b_ub[row_idx]=0

    # 3) Bounds for x_{a,h}
    #    If hour h in intervals => [pmin_a, pmax_a]
    #    Else => (0, 0)
    # plus L >= 0
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
                bounds.append((0.0, 0.0))

    # bound for L => (0, None)
    bounds.append((0, None))

    # Solve
    res = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method='highs'
    )
    if not res.success:
        print("linprog failed:", res.message)
        return None, None, None

    # parse
    x_solution = res.x
    L_val = x_solution[L_index]
    total_obj = res.fun

    # reconstruct x_dict
    x_dict = {}
    for a_idx, a_name in enumerate(A_all):
        x_dict[a_name] = {}
        for h in range(1, nH+1):
            x_dict[a_name][h] = x_solution[var_idx(a_idx,h)]

    return x_dict, L_val, total_obj


def plot_price_curve(prices):
    """
    Generate a plot of the hourly electricity price trend for the NO1 region.

    Parameters:
    - prices (dict): A dictionary with hourly electricity prices in NOK/kWh.

    Returns:
    None. This function displays the price trend plot.
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

def plot_usage_stacked(x_dict, appliances_dict):
    """
    Create a stacked bar chart to visualize scheduled energy usage by hour for each appliance.

    Parameters:
    - x_dict (dict): Hourly energy usage data for each appliance.
    - appliances_dict (dict): Appliance data providing context, like color codes.

    Returns:
    None. Displays a stacked bar chart showing hourly usage across appliances.
    """
    hours = list(range(1,25))
    bottom = [0.0]*24
    plt.figure(figsize=(12,6))
    for a_name in appliances_dict:
        usage_a = [x_dict[a_name][h] for h in hours]
        plt.bar(hours, usage_a, bottom=bottom,
                color=appliances_dict[a_name]["color"],
                label=a_name)
        for i in range(24):
            bottom[i] += usage_a[i]

    plt.title("Scheduled Energy Usage by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Usage (kWh)")
    plt.xticks(hours)
    plt.ylim(0, 5)
    plt.legend(ncol=2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_hourly_cost(x_dict, appliances_dict, prices):
    """
    Plot a bar chart representing hourly energy costs based on appliance schedules.

    Parameters:
    - x_dict (dict): Hourly energy usage for each appliance.
    - appliances_dict (dict): Appliance data for context.
    - prices (dict): Hourly electricity prices used for cost calculation.

    Returns:
    None. Displays the bar chart of hourly electricity costs and prints the total daily cost.
    """
    hours = range(1,25)
    cost_hourly = []
    for h in hours:
        usage_sum = sum(x_dict[a_name][h] for a_name in appliances_dict)
        cost_hourly.append(usage_sum * prices[h])
    total_cost = sum(cost_hourly)

    plt.figure(figsize=(10,4))
    plt.bar(hours, cost_hourly, color="skyblue", edgecolor="black")
    plt.title(f"Hourly Cost (Total={total_cost:.2f} NOK)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Cost (NOK)")
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print(f"Total daily cost (recomputed from hours): {total_cost:.2f} NOK")

def main():
    """
    Execute the main workflow for appliance energy scheduling and cost optimization.

    Steps involved:
    - Prepare appliance power constraints from raw data.
    - Fetch current electricity prices or apply a fallback rate.
    - Solve the scheduling problem with an added peak load variable to optimize both energy usage and peak load.
    - Print usage details and cost analysis.
    - Generate plots to visually present prices, energy usage, and costs.

    Returns:
    None. Manages the entire sequence of operations and displays results through plots and console outputs.
    """
    # 1) Build pmin/pmax from the simplified rules
    final_appliances = build_pmin_pmax_dict(appliances_raw)

    # 2) Fetch real-time prices (same as Task 2) or fallback
    try:
        prices = fetch_no1_prices_for_today()
    except Exception as e:
        print("Warning: fallback due to:", e)
        prices = {h:1.0 for h in range(1,25)}

    # 3) Solve with new peak-load approach
    alpha = 1.0  # weighting for L in the objective
    x_dict, L_val, total_obj = solve_scheduling_lp_with_peak(prices, final_appliances, alpha)
    if x_dict is None:
        print("No feasible solution, aborting.")
        return

    print("\nlinprog: Optimal solution found (with peak-load constraint).")
    print(f"Objective Value = {total_obj:.2f} (Energy Cost + alpha * L)")
    print(f"Peak Load L = {L_val:.3f} kW\n")

    # 4) Compute pure energy cost separate from the objective
    energy_cost = 0.0
    for h in range(1,25):
        usage_sum = sum(x_dict[a_name][h] for a_name in x_dict)
        energy_cost += usage_sum * prices[h]

    print(f"Energy cost portion: {energy_cost:.2f} NOK")
    print(f"Peak penalty portion: alpha * L = {alpha} * {L_val:.3f} = {alpha * L_val:.2f}\n")

    # 5) Print usage details
    for a_name in final_appliances:
        E_req = final_appliances[a_name]["E"]
        used = sum(x_dict[a_name][h] for h in range(1,25))
        print(f"Appliance: {a_name}, usage={used:.4f} kWh (target={E_req})")
        for h in range(1,25):
            val = x_dict[a_name][h]
            if abs(val)>1e-9:
                print(f"  Hour {h}: {val:.4f}")
        print()

    # 6) Plots
    plot_price_curve(prices)
    plot_usage_stacked(x_dict, final_appliances)
    plot_hourly_cost(x_dict, final_appliances, prices)

if __name__ == "__main__":
    main()