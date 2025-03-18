from scipy.optimize import linprog
from appliances_2_4 import appliances_raw
import requests
import datetime
import numpy as np
import matplotlib.pyplot as plt

def build_hour_usage_constraints(appliances):
    constraints = {}
    for a_name, appliance in appliances.items():
        intervals = appliance["intervals"]
        E_total = appliance["E"]
        must_run = appliance["must_run"]
        gamma_max = appliance["gamma_max"]

        if must_run:
            # Divide total energy evenly across allowed intervals
            hour_count = sum(end - start + 1 for start, end in intervals)
            rate = E_total / hour_count

            for start, end in intervals:
                for hour in range(start, end + 1):
                    constraints.setdefault(hour, []).append((a_name, rate if rate <= gamma_max else gamma_max))
        else:
            for start, end in intervals:
                for hour in range(start, end + 1):
                    constraints.setdefault(hour, []).append((a_name, (0, gamma_max)))  # min 0, max gamma_max

    return constraints

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
        hour_idx = i + 1
        prices[hour_idx] = entry["NOK_per_kWh"]
    return prices

def solve_scheduling_lp(prices, hour_constraints, peak_load):
    total_vars = sum(len(hour_constraints[h]) for h in range(1, 25))
    nH = 24

    # Objective: Minimize cost
    c = []
    bounds = []
    indices = {}
    idx = 0

    for h in range(1, nH + 1):
        hourly_constraints = hour_constraints.get(h, [])
        for appliance, constraint in hourly_constraints:
            if isinstance(constraint, tuple):  # shiftable appliance
                pmin_a, pmax_a = constraint
            else:  # must run
                pmin_a = constraint
                pmax_a = constraint
            
            c.append(prices[h])
            bounds.append((pmin_a, pmax_a))
            indices[(h, appliance)] = idx
            idx += 1

    # Usage constraints: Ensure each appliance meets its total energy demand
    A_eq = []
    b_eq = []

    for appliance, data in appliances_raw.items():
        E_total = data["E"]
        row = [0] * total_vars
        for h in range(1, nH + 1):
            if (h, appliance) in indices:
                row[indices[(h, appliance)]] = 1.0
        A_eq.append(row)
        b_eq.append(E_total)

    # Peak load constraints
    A_ub = []
    b_ub = []

    for h in range(1, nH + 1):
        row = [0] * total_vars
        for appliance in appliances_raw:
            if (h, appliance) in indices:
                row[indices[(h, appliance)]] = 1.0
        A_ub.append(row)
        b_ub.append(peak_load[h])

    # Solve
    res = linprog(
        c=c,
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method='highs'
    )

    if not res.success:
        print("linprog failed:", res.message)
        return None, None

    # Parse solution
    x_dict = {a_name: {h: 0 for h in range(1, 25)} for a_name in appliances_raw}
    for (h, appliance), idx in indices.items():
        x_dict[appliance][h] = res.x[idx]

    return x_dict, res.fun

def plot_price_curve(prices):
    hours = range(1, 25)
    vals = [prices[h] for h in hours]
    plt.figure(figsize=(8, 4))
    plt.plot(hours, vals, marker='o')
    plt.title("Hourly Electricity Price (NO1)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Price (NOK/kWh)")
    plt.xticks(hours)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_usage_stacked(x_dict, appliances_dict):
    hours = list(range(1, 25))
    bottom = [0.0] * 24
    plt.figure(figsize=(12, 6))

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
    hours = range(1, 25)
    cost_hourly = []
    for h in hours:
        usage_sum = sum(x_dict[a_name][h] for a_name in appliances_dict)
        cost_hourly.append(usage_sum * prices[h])
    total_cost = sum(cost_hourly)

    plt.figure(figsize=(10, 4))
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
    # 1) Build hourly usage constraints:
    hour_constraints = build_hour_usage_constraints(appliances_raw)

    # 2) Prices
    try:
        prices = fetch_no1_prices_for_today()
    except Exception as e:
        print("Warning: fallback due to:", e)
        prices = {h: 1.0 for h in range(1, 25)}

    # 3) Define peak loads for each hour (example: constant peak load of 5 kW for each hour)
    peak_load = {h: 3.0 for h in range(1, 25)}  # You can customize this dictionary

    # 4) Solve
    x_dict, total_cost = solve_scheduling_lp(prices, hour_constraints, peak_load)
    if x_dict is None:
        print("No feasible solution, aborting.")
        return

    print("\nlinprog: Optimal solution found")
    print(f"Optimal daily cost from solver: {total_cost:.2f} NOK\n")

    # 5) Print usage
    for a_name in appliances_raw:
        E_req = appliances_raw[a_name]["E"]
        used = sum(x_dict[a_name][h] for h in range(1, 25))
        print(f"Appliance: {a_name}, usage={used:.4f} kWh (target={E_req})")
        for h in range(1, 25):
            val = x_dict[a_name][h]
            if abs(val) > 1e-9:
                print(f"  Hour {h}: {val:.4f}")
        print()

    # 6) Plots
    plot_price_curve(prices)
    plot_usage_stacked(x_dict, appliances_raw)
    plot_hourly_cost(x_dict, appliances_raw, prices)

if __name__ == "__main__":
    main()
