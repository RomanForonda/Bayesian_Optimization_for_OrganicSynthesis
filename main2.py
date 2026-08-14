#Using categorical parameters (Solvent) + continuous parameters (Temperature and Time) to plot the landscape and the optimization progress.
#Includes a simulated experiment function to generate yield values based on the parameters.

from baybe import Campaign
from baybe.parameters import (NumericalContinuousParameter, CategoricalParameter)
from baybe.searchspace import SearchSpace
from baybe.objectives import SingleTargetObjective
from baybe.targets import NumericalTarget

import matplotlib.pyplot as plt
import numpy as np

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
    values=["DCM", "MeOH", "ACN"]
)

# ==========================================
# 2. Define the search space
# ==========================================

searchspace = SearchSpace.from_product(
    parameters=[temperature, time, solvent]
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

def simulate_experiment(temperature, time, solvent):

    solvent_effects = {
        "DCM": 5,
        "MeOH": -10,
        "ACN": 2,
    }

    yield_value = (
        100
        - 0.02 * (temperature - 100) ** 2
        - 0.5 * (time - 12) ** 2
        + solvent_effects[solvent]
    )

    noise = np.random.normal(0, 3)

    return min(100, max(0, yield_value + noise))

# ==========================================
# 7. Store experimental results
# ==========================================

results = []


# ==========================================
# 8. Bayesian Optimization loop
# ==========================================

for i in range(20):

    # Ask BayBE for the next experiment
    recommendation = campaign.recommend(batch_size=1)

    temperature_value = recommendation["Temperature"].iloc[0]
    time_value = recommendation["Time"].iloc[0]
    solvent_value = recommendation["Solvent"].iloc[0]

    # Simulate experiment
    yield_value = simulate_experiment(
        temperature_value,
        time_value,
        solvent_value
    )

    # Give result to BayBE
    recommendation["Yield"] = yield_value

    campaign.add_measurements(recommendation)

    # Store result
    results.append({
        "Experiment": i + 1,
        "Temperature": temperature_value,
        "Time": time_value,
        "Solvent": solvent_value,
        "Yield": yield_value,
    })

    print(
        f"Experiment {i + 1:02d}: "
        f"Temperature = {temperature_value:.2f} °C, "
        f"Time = {time_value:.2f} h, "
        f"Solvent = {solvent_value}, "
        f"Yield = {yield_value:.2f}%"
    )

print("\n================ CAMPAIGN ================\n")
print(campaign)

