#Using continuous parameters only (Temperature and Time) to plot the landscape and the optimization progress.
#Includes a simulated experiment function to generate yield values based on the parameters.

from baybe import Campaign
from baybe.parameters import NumericalContinuousParameter
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

# ==========================================
# 2. Define the search space
# ==========================================

searchspace = SearchSpace.from_product(
    parameters=[temperature, time]
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

def simulate_experiment(temperature, time):

    noise = np.random.normal(0, 3)    #Anadimos ruido experimental
    yield_value = (
        100
        - 0.02 * (temperature - 100) ** 2
        - 0.5 * (time - 12) ** 2
        + noise
    )

    return max(0, yield_value)

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

    # Simulate experiment
    yield_value = simulate_experiment(
        temperature_value,
        time_value,
    )

    # Give result to BayBE
    recommendation["Yield"] = yield_value

    campaign.add_measurements(recommendation)

    # Store result
    results.append({
        "Experiment": i + 1,
        "Temperature": temperature_value,
        "Time": time_value,
        "Yield": yield_value,
    })

    print(
        f"Experiment {i + 1:02d}: "
        f"Temperature = {temperature_value:.2f} °C, "
        f"Time = {time_value:.2f} h, "
        f"Yield = {yield_value:.2f}%"
    )

print("\n================ CAMPAIGN ================\n")
print(campaign)

# ==========================================
# 9. Create the true Yield landscape
# ==========================================

temperatures = np.linspace(20, 150, 200)
times = np.linspace(1, 24, 200)

T, TIME = np.meshgrid(temperatures, times)

YIELD = (
    100
    - 0.02 * (T - 100) ** 2
    - 0.5 * (TIME - 12) ** 2
)

YIELD = np.maximum(0, YIELD)


# ==========================================
# 10. Plot the landscape
# ==========================================

plt.figure(figsize=(10, 7))

contour = plt.contourf(
    T,
    TIME,
    YIELD,
    levels=30,
)

plt.colorbar(contour, label="Yield (%)")


# ==========================================
# 11. Plot BayBE experiments
# ==========================================

experiment_temperatures = [
    result["Temperature"]
    for result in results
]

experiment_times = [
    result["Time"]
    for result in results
]

plt.scatter(
    experiment_temperatures,
    experiment_times,
    s=60,
    edgecolor="black",
    label="BayBE experiments",
)


# ==========================================
# 12. Mark the true optimum
# ==========================================

plt.scatter(
    100,
    12,
    marker="*",
    s=250,
    edgecolor="black",
    label="True optimum",
)


# ==========================================
# 13. Labels
# ==========================================

plt.xlabel("Temperature (°C)")
plt.ylabel("Time (h)")

plt.title("Bayesian Optimization of Temperature and Time")

plt.legend()

plt.tight_layout()

plt.show()

# ==========================================
# 14. Best Yield found over time
# ==========================================

experiment_numbers = [
    result["Experiment"]
    for result in results
]

yields = [
    result["Yield"]
    for result in results
]

best_yields = np.maximum.accumulate(yields)


# ==========================================
# 15. Plot optimization progress
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    experiment_numbers,
    best_yields,
    marker="o",
)

plt.xlabel("Experiment")
plt.ylabel("Best Yield found so far (%)")

plt.title("Bayesian Optimization Progress")

plt.grid(True)

plt.tight_layout()

plt.show()