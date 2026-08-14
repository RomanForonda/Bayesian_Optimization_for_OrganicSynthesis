#Introducing interaction between continuous and categorical parameters (Temperature, Time, and Solvent) to plot the landscape and the optimization progress.
#Includes a simulated experiment function to generate yield values based on the parameters.
#Includes constrains (DCM → T ≤ 40 °C) and boundaries (T > 130 °C → degradación) on the search space to avoid unfeasible conditions.

from baybe import Campaign
from baybe.parameters import (NumericalContinuousParameter, CategoricalParameter)
from baybe.searchspace import SearchSpace
from baybe.objectives import SingleTargetObjective
from baybe.targets import NumericalTarget
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. Define experimental parameters
# ==========================================

temperature = NumericalContinuousParameter(
    name="Temperature",
    bounds=(20, 150),
)

time = NumericalContinuousParameter(
    name="Time",
    bounds=(1, 24),
)

solvent = CategoricalParameter(
    name="Solvent",
    values=["DCM", "ACN", "MeOH"]
)

base = CategoricalParameter(
    name="Base",
    values=["Et3N", "DIPEA", "K2CO3"]
)

# ==========================================
# 2. Define the search space
# ==========================================

searchspace = SearchSpace.from_product(
    parameters=[temperature, time, solvent, base]
)

# ==========================================
# 3. Target
# ==========================================

target = NumericalTarget(
    name="Yield",
    mode="MAX",
)

# ==========================================
# 4. Objective
# ==========================================

objective = SingleTargetObjective(
    target=target,
)

# ==========================================
# 5. Campaign
# ==========================================

campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
)

# ==========================================
# 6. Simulated experiment
# ==========================================

def simulate_experiment(temperature, time, solvent, base):

    solvent_base_effects = {

        ("DCM", "Et3N"): 5,
        ("DCM", "DIPEA"): 2,
        ("DCM", "K2CO3"): -5,

        ("ACN", "Et3N"): 1,
        ("ACN", "DIPEA"): 4,
        ("ACN", "K2CO3"): 3,

        ("MeOH", "Et3N"): -5,
        ("MeOH", "DIPEA"): -8,
        ("MeOH", "K2CO3"): -12,
    }

    yield_value = (
        100
        - 0.02 * (temperature - 100) ** 2
        - 0.5 * (time - 12) ** 2
        + solvent_base_effects[(solvent, base)]
    )

    noise = np.random.normal(0, 3)

    return min(100, max(0, yield_value + noise))

# ==========================================
# 7. Store experimental results
# ==========================================

results = []

# ==========================================
# 8. SCOUTING PHASE
# ==========================================

scouting_conditions = list(
    itertools.product(
        ["DCM", "ACN", "MeOH"],
        ["Et3N", "DIPEA", "K2CO3"],
    )
)

print("\n================ SCOUTING ================\n")

for i, (solvent_value, base_value) in enumerate(
    scouting_conditions
):

    # Random temperature and time
    temperature_value = np.random.uniform(20, 150)
    time_value = np.random.uniform(1, 24)

    # Simulate experiment
    yield_value = simulate_experiment(
        temperature_value,
        time_value,
        solvent_value,
        base_value,
    )

    # Create measurement dataframe
    measurement = pd.DataFrame({
        "Temperature": [temperature_value],
        "Time": [time_value],
        "Solvent": [solvent_value],
        "Base": [base_value],
        "Yield": [yield_value],
    })

    # Give scouting result to BayBE
    campaign.add_measurements(measurement)

    # Store result
    results.append({
        "Experiment": i + 1,
        "Phase": "Scouting",
        "Temperature": temperature_value,
        "Time": time_value,
        "Solvent": solvent_value,
        "Base": base_value,
        "Yield": yield_value,
    })

    print(
        f"Scouting {i + 1:02d}: "
        f"Temperature = {temperature_value:.2f} °C, "
        f"Time = {time_value:.2f} h, "
        f"Solvent = {solvent_value}, "
        f"Base = {base_value}, "
        f"Yield = {yield_value:.2f}%"
    )

print("\n================ SCOUTING RESULTS ================\n")

scouting_df = pd.DataFrame(results[:9])

print(
    scouting_df[
        [
            "Temperature",
            "Time",
            "Solvent",
            "Base",
            "Yield",
        ]
    ].sort_values(
        "Yield",
        ascending=False,
    )
)
# ==========================================
# 9. Bayesian Optimization PHASE
# ==========================================

for i in range(20):

    # Ask BayBE for the next experiment
    recommendation = campaign.recommend(batch_size=1)

    temperature_value = recommendation["Temperature"].iloc[0]
    time_value = recommendation["Time"].iloc[0]
    solvent_value = recommendation["Solvent"].iloc[0]
    base_value = recommendation["Base"].iloc[0]

    # Simulate experiment
    yield_value = simulate_experiment(
        temperature_value,
        time_value,
        solvent_value,
        base_value
    )

    # Give result to BayBE
    recommendation["Yield"] = yield_value

    campaign.add_measurements(recommendation)

    # Store result
    results.append({
        "Experiment": i + 10,
        "Phase": "Bayesian Optimization",
        "Temperature": temperature_value,
        "Time": time_value,
        "Solvent": solvent_value,
        "Base": base_value,
        "Yield": yield_value,
    })

    print(
        f"BO {i + 1:02d}: "
        f"Temperature = {temperature_value:.2f} °C, "
        f"Time = {time_value:.2f} h, "
        f"Solvent = {solvent_value}, "
        f"Base = {base_value}, "
        f"Yield = {yield_value:.2f}%"
    )

print("\n================ CAMPAIGN ================\n")
print(campaign)

