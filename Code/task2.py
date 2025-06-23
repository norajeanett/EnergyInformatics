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


"""
- beskrivelse
- params
- return
- raises 
"""

def forced_usage(e_total, hour_count):
    """
    Compute forced kWh/hour for a must-run device.
    Parameters:
    - e_total (float): Total energy consumption of an appliance
    - hour_count (int): How many hours an appliance is allowed to run

    returns:
    - float: The average energy consumption per hour (kWh/hour) for the must-run appliance. 
      Returns 0.0 if hour_count is less than 1, indicating that no valid hourly distribution can be made.

    """
    if hour_count < 1:
        return 0.0
    return e_total / hour_count

def build_pmin_pmax_dict(appliances_raw):
   
    """
    Build a dictionary specifying the minimum and maximum power usage for each appliance per hour.

    Parameters:
    - appliances_raw (dict): A dictionary containing information about each appliance.
      Each entry should include:
        - 'E' (float): Total energy requirement (kWh).
        - 'intervals' (list of tuples): Time intervals during which the appliance must or can operate.
        - 'gamma_max' (float): Maximum allowable power (kW) in any hour.
        - 'must_run' (bool): Flag indicating if the appliance must run strictly according to its energy requirement.
        - 'color' (str): A string representing a color code for visualization purposes.

    Returns:
    - dict: A dictionary where each key is an appliance name, and the value is another dictionary including:
        - 'E': Total energy requirement (kWh).
        - 'intervals': Operating intervals for the appliance.
        - 'pmin': Minimum power usage per hour (kW).
        - 'pmax': Maximum power usage per hour (kW).
        - 'color': Visual color indicator.

    Behavior:
    - If an appliance has 'must_run' set to True, calculates 'pmin' and 'pmax' as the total energy divided by the number of operational hours.
    - If 'forced usage' exceeds 'gamma_max', a warning is printed indicating potential infeasibility.
    - For appliances with 'must_run' set to False, sets 'pmin' to 0 and 'pmax' to 'gamma_max', indicating flexible scheduling.

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
            # forced usage
            rate = forced_usage(E_a, hour_count)
            pmin_val = rate
            pmax_val = rate

            # If forced usage > gamma_max, might STILL be infeasible. We'll warn user:
            if rate > gamma_max + 1e-9:
                print(f"WARNING: {a_name}: forced usage={rate:.4f} kW > gamma_max={gamma_max:.4f}")
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
    Fetch electricity prices for the NO1 region for today.

    This function retrieves electricity price data from the 'Hva koster strømmen' API for the current date
    and reads prices in NOK/kWh for each hour in the NO1 region.

    Returns:
    - dict: A dictionary where each key is an hour index (1-24) and the value is the electricity price (float)
            in NOK/kWh for that hour.

    Raises:
    - requests.exceptions.HTTPError: If the HTTP request to fetch the data fails, an exception is raised
      indicating that the request returned an unsuccessful status code.
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
    Solve the linear programming problem for scheduling appliance usage to minimize electricity costs.

    Objective:
    MINIMIZE: sum_{h=1..24} price[h] * sum_{a} x_{a,h}
    This objective calculates the total electricity cost across all appliances and hours.

    Subject to constraints:
    - sum_{h} x_{a,h} = E_a for each appliance a.
      Each appliance must consume its total required energy 'E' over the scheduled hours.
      
    - x_{a,h} = 0 if the hour 'h' is not within the operating intervals for appliance a.
      Appliances must have zero consumption outside their designated operational intervals.

    - x_{a,h} in [pmin_a, pmax_a] if hour 'h' is within the intervals for appliance a.
      Applies minimum and maximum power bounds during valid intervals for each appliance.

    Parameters:
    - prices (dict): A dictionary where keys are hour indices (1-24) and values are electricity prices in NOK/kWh.
    - appliances_dict (dict): A dictionary containing appliance data. Each key is the appliance name, and the value is another dictionary with:
      - 'E' (float): Total energy requirement (kWh) for the appliance.
      - 'intervals' (list of tuples): List of hour intervals (start, end) during which the appliance can operate.
      - 'pmin' (float): Minimum power usage per hour (kW).
      - 'pmax' (float): Maximum power usage per hour (kW).

    Returns:
    - dict: A dictionary mapping each appliance name to another dictionary that maps each hour to its scheduled power usage (x_{a,h} in kW).
    - float: The minimized total electricity cost (objective value).

    Notes:
    - If the linear programming solver fails, prints a message and returns (None, None).
    - Uses the 'highs' method in the `linprog` function from SciPy for optimization.

    """

    A_all = list(appliances_dict.keys())
    nA = len(A_all)
    nH = 24
    total_vars = nA*nH

    def var_idx(a_idx, hour):
        return a_idx*nH + (hour-1)

    # 1) Objective
    c = np.zeros(total_vars)
    for a_idx, a_name in enumerate(A_all):
        for h in range(1, nH+1):
            c[var_idx(a_idx, h)] = prices[h]

    # 2) sum_{h} x_{a,h} = E_a
    A_eq = np.zeros((nA, total_vars))
    b_eq = np.zeros(nA)
    for a_idx, a_name in enumerate(A_all):
        E_a = appliances_dict[a_name]["E"]
        for h in range(1, nH+1):
            A_eq[a_idx, var_idx(a_idx,h)] = 1.0
        b_eq[a_idx] = E_a

    # 3) Bounds
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

    # 4) Solve
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
    # parse
    x_dict = {}
    for a_idx, a_name in enumerate(A_all):
        x_dict[a_name] = {}
        for h in range(1, nH+1):
            x_dict[a_name][h] = x_solution[var_idx(a_idx, h)]
    return x_dict, res.fun

def plot_price_curve(prices):
    """
    Plot the hourly electricity price curve (for the NO1 region.)

    This function generates a line plot representing electricity prices for each hour of the day, visualized 
    with markers to highlight individual data points.

    Parameters:
    - prices (dict): A dictionary where keys are hour indices (1-24) and values are electricity prices in NOK/kWh.

    Returns:
    None. Displays the plot of hourly electricity prices.

    Plot Details:
    - X-axis: Hour of the day (1 to 24).
    - Y-axis: Price of electricity in NOK/kWh.
    - The plot includes a title, axis labels, grid lines, and a tight layout for clarity.
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
    Plot a stacked bar chart of scheduled energy usage by hour for various appliances.

    This function visualizes the hourly energy consumption for each appliance in a stacked format, 
    allowing for a clear comparison of how energy usage accumulates across appliances throughout the day.

    Parameters:
    - x_dict (dict): A dictionary mapping each appliance name to another dictionary that maps each hour 
                     to its scheduled power usage (kWh).
    - appliances_dict (dict): A dictionary containing appliance data. Each entry includes:
      - 'color' (str): A color code for the appliance used in the plot.

    Returns:
    None. Displays a stacked bar chart showing energy usage by hour for each appliance.

    Plot Details:
    - X-axis: Hour of the day (1 to 24).
    - Y-axis: Energy usage in kWh.
    - The chart includes a legend with appliance names, grid lines for the y-axis, a title, and axis labels.
    - Utilizes different colors for each appliance for visual differentiation, as specified in `appliances_dict`.

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
    plt.legend(ncol=2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_hourly_cost(x_dict, appliances_dict, prices):
    """
    Plot a bar chart of hourly energy costs based on scheduled appliance usage.

    This function calculates and visualizes the cost of electricity for each hour of the day, taking into account
    the scheduled energy usage of various appliances and hourly electricity prices. It also computes the total daily cost.

    Parameters:
    - x_dict (dict): A dictionary mapping each appliance name to another dictionary, which maps each hour 
                     to its scheduled power usage (kWh).
    - appliances_dict (dict): A dictionary containing appliance data.
    - prices (dict): A dictionary where keys are hour indices (1-24) and values are electricity prices in NOK/kWh.

    Returns:
    None. Displays a bar chart showing the cost of electricity for each hour and prints the total daily cost.

    Plot Details:
    - X-axis: Hour of the day (1 to 24).
    - Y-axis: Cost of electricity in NOK.
    - Each bar represents the cost for a specific hour, calculated as the sum of all appliance usage for that hour multiplied by the corresponding price.
    - The chart includes a title with total daily cost, grid lines for the y-axis, axis labels, and uses soft colors for clear visual presentation.

    Notes:
    - The total cost is printed to the console for convenience, reflecting a summation of hourly costs.
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
    Execute the main workflow for appliance scheduling and cost optimization.

    This function performs the following steps:
    1. Builds minimum and maximum power constraints for each appliance from simplified rules.
    2. Fetches current day's electricity prices for the NO1 region and handles potential exceptions with a fallback mechanism.
    3. Solves the linear programming problem to schedule appliances and minimize costs; checks for feasible solutions.
    4. Prints the optimal daily cost and individual appliance usage against their energy targets.
    5. Generates plots to visualize price trends, scheduled energy usage, and hourly cost distribution.

    Returns:
    None. This function serves as the main entry point, performing operations sequentially and displaying results through print statements and plots.

    Notes:
    - Implements error handling for data fetching, using default prices on failure.
    - Utilizes visualization functions to present comprehensive usage data and cost analysis.
    """

    # 1) Build pmin/pmax from the simplified rules:
    final_appliances = build_pmin_pmax_dict(appliances_raw)

    # 2) Prices
    try:
        prices = fetch_no1_prices_for_today()
    except Exception as e:
        print("Warning: fallback due to:", e)
        prices = {h:1.0 for h in range(1,25)}

    # 3) Solve
    x_dict, total_cost = solve_scheduling_lp(prices, final_appliances)
    if x_dict is None:
        print("No feasible solution, aborting.")
        return

    print("\nlinprog: Optimal solution found")
    print(f"Optimal daily cost from solver: {total_cost:.2f} NOK\n")

    # 4) Print usage
    for a_name in final_appliances:
        E_req = final_appliances[a_name]["E"]
        used = sum(x_dict[a_name][h] for h in range(1,25))
        print(f"Appliance: {a_name}, usage={used:.4f} kWh (target={E_req})")
        for h in range(1,25):
            val = x_dict[a_name][h]
            if abs(val)>1e-9:
                print(f"  Hour {h}: {val:.4f}")
        print()

    # 5) Plots
    plot_price_curve(prices)
    plot_usage_stacked(x_dict, final_appliances)
    plot_hourly_cost(x_dict, final_appliances, prices)

if __name__ == "__main__":
    main()
