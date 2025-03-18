import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Define constants
HOURS_PER_DAY = 24  # Total hours in a day
PEAK_HOURS = range(17, 20)  # Peak electricity hours
PEAK_COST = 1.0  # Cost per kWh during peak hours
NON_PEAK_COST = 0.5  # Cost per kWh during non-peak hours

# Create an array representing electricity costs for each hour
electricity_costs = np.array([PEAK_COST if hour in PEAK_HOURS else NON_PEAK_COST for hour in range(HOURS_PER_DAY)])

# Define appliances with their energy requirements, max power, and allowed operation intervals
appliance_data = {
    "Washing machine": {
        "energy_required": 1.94,
        "max_power": 1.4,
        "allowed_intervals": [(16, 22)],
    },
    "Electric vehicle": {
        "energy_required": 9.90,
        "max_power": 3.6,
        "allowed_intervals": [(0, 5), (6, 8), (16, 23)],
    },
    "Dish washer": {
        "energy_required": 1.44,
        "max_power": 1.2,
        "allowed_intervals": [(16, 22)],
    }
}

num_appliances = len(appliance_data)
total_variables = num_appliances * HOURS_PER_DAY

cost_coefficients = np.tile(electricity_costs, num_appliances)

# Constraint: Each appliance must consume exactly its required energy
energy_constraints_matrix = np.zeros((num_appliances, total_variables))
energy_constraints_rhs = np.zeros(num_appliances)

for i, (appliance, parameters) in enumerate(appliance_data.items()):
    energy_constraints_matrix[i, i * HOURS_PER_DAY:(i + 1) * HOURS_PER_DAY] = 1
    energy_constraints_rhs[i] = parameters['energy_required']

# Constraint: Max power limit per hour per appliance
upper_bound_matrix = []
upper_bound_rhs = []

for i, (appliance, parameters) in enumerate(appliance_data.items()):
    for hour in range(HOURS_PER_DAY):
        idx = i * HOURS_PER_DAY + hour
        allowed_hours = any(interval[0] <= hour <= interval[1] for interval in parameters['allowed_intervals'])

        constraint_row = np.zeros(total_variables)
        constraint_row[idx] = 1

        if allowed_hours:
            upper_bound_matrix.append(constraint_row)
            upper_bound_rhs.append(parameters['max_power'])
        else:
            upper_bound_matrix.append(constraint_row)
            upper_bound_rhs.append(0)  # Setter effektforbruk til 0 utenfor intervallet

# Constraint: Total power used at any hour should not exceed 3.6 kW
total_power_constraints_matrix = np.zeros((HOURS_PER_DAY, total_variables))
total_power_constraints_rhs = np.full(HOURS_PER_DAY, 3.6)

for hour in range(HOURS_PER_DAY):
    for i in range(num_appliances):
        total_power_constraints_matrix[hour, i * HOURS_PER_DAY + hour] = 1

upper_bound_matrix.extend(total_power_constraints_matrix)
upper_bound_rhs.extend(total_power_constraints_rhs)

# Convert constraints to numpy arrays
upper_bound_matrix = np.array(upper_bound_matrix)
upper_bound_rhs = np.array(upper_bound_rhs)

# Solve the linear programming problem
result = linprog(cost_coefficients, A_eq=energy_constraints_matrix, b_eq=energy_constraints_rhs,
                 A_ub=upper_bound_matrix, b_ub=upper_bound_rhs, bounds=(0, None), method='highs')

if result.success:
    print(f"Optimal total cost: {result.fun:.2f} NOK")

    # Print actual energy consumption for each appliance
    optimal_schedule = result.x.reshape(num_appliances, HOURS_PER_DAY)
    for i, (appliance, parameters) in enumerate(appliance_data.items()):
        total_usage = sum(optimal_schedule[i])
        print(f"{appliance} total usage: {total_usage:.2f} kWh (Required: {parameters['energy_required']} kWh)")
else:
    print("Optimization failed!")

# Reshape result into a schedule for each appliance
optimal_schedule = result.x.reshape(num_appliances, HOURS_PER_DAY)

# Plot as a stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(HOURS_PER_DAY)

bottom = np.zeros(HOURS_PER_DAY)
colors = ['lightblue', 'lightgreen', 'lightgrey']  # Assign colors to appliances

for i, (appliance, _) in enumerate(appliance_data.items()):
    ax.bar(x, optimal_schedule[i], label=appliance, bottom=bottom, color=colors[i % len(colors)])
    bottom += optimal_schedule[i]

# Highlight peak hours with a background shading limited to peak cost level
for hour in PEAK_HOURS:
    ax.fill_between([hour, hour + 1], 0, PEAK_COST, color='red', alpha=0.2, label='Peak Hours' if hour == 17 else "")

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Usage (kWh)")
ax.set_title("Scheduled Energy Usage by Hour with Peak Hours Highlighted")
ax.legend()
plt.xticks(range(HOURS_PER_DAY))
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()