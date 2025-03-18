appliances_raw = {
    "Lighting": {
        "E": 2.0,
        "intervals": [(11, 20)],
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

    # SHIFTABLE APPLIANCES
    "ElectricStove": {
        "E": 3.9,
        "intervals": [(16, 18)],
        "gamma_max": 5.0,   # can go up to 5kW
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
    "EV": {
        "E": 9.90,
        "intervals": [(6, 8), (16, 23)],
        "gamma_max": 7.2,
        "must_run": False,
        "color": "gray"
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