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
    """Compute forced kWh/hour for a must-run device."""
    if hour_count < 1:
        return 0.0
    return e_total / hour_count

def build_pmin_pmax_dict(appliances_raw):
    """
    If must_run=True => pmin=pmax= E/(# hours).
    If must_run=False => pmin=0, pmax= gamma_max.
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
    MINIMIZE: sum_{h=1..24} price[h]* sum_{a} x_{a,h}
    subject to:
       sum_{h} x_{a,h} = E_a
       x_{a,h} = 0 if hour not in intervals
       x_{a,h} in [pmin_a, pmax_a] if hour in intervals
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
