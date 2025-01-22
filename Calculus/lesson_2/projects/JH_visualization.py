import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# Load the CSV
file_path = "john_hohman_project_1_results.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Extract columns
time = data["Time"]
F1_actual = data["F1_Actual"]
F1_approx = data["F1_Approximation"]
F1_error = data["F1_RelativeError"]

F2_actual = data["F2_Actual"]
F2_approx = data["F2_Approximation"]
F2_error = data["F2_RelativeError"]

F3_actual = data["F3_Actual"]
F3_approx = data["F3_Approximation"]
F3_error = data["F3_RelativeError"]

# Create separate figures for each function
fig1, ax1 = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1])
fig2, ax2 = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1])
fig3, ax3 = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1])

# Plot F1
ax1[0].plot(time, F1_actual, label="F1 Actual", color="blue", linewidth=2)
ax1[0].plot(time, F1_approx, label="F1 Approximation", color="blue", linestyle="--", linewidth=2)
ax1[0].set_title("F1: Actual vs Approximation")
ax1[0].set_xlabel("Time")
ax1[0].set_ylabel("Values")
ax1[0].legend()
ax1[0].grid()

ax1[1].plot(time, F1_error, label="F1 Relative Error", color="blue", linewidth=2)
ax1[1].set_title("F1: Relative Error")
ax1[1].set_xlabel("Time")
ax1[1].set_ylabel("Relative Error")
ax1[1].legend()
ax1[1].grid()

# Plot F2
ax2[0].plot(time, F2_actual, label="F2 Actual", color="orange", linewidth=2)
ax2[0].plot(time, F2_approx, label="F2 Approximation", color="orange", linestyle="--", linewidth=2)
ax2[0].set_title("F2: Actual vs Approximation")
ax2[0].set_xlabel("Time")
ax2[0].set_ylabel("Values")
ax2[0].legend()
ax2[0].grid()

ax2[1].plot(time, F2_error, label="F2 Relative Error", color="orange", linewidth=2)
ax2[1].set_title("F2: Relative Error")
ax2[1].set_xlabel("Time")
ax2[1].set_ylabel("Relative Error")
ax2[1].legend()
ax2[1].grid()

# Plot F3
ax3[0].plot(time, F3_actual, label="F3 Actual", color="green", linewidth=2)
ax3[0].plot(time, F3_approx, label="F3 Approximation", color="green", linestyle="--", linewidth=2)
ax3[0].set_title("F3: Actual vs Approximation")
ax3[0].set_xlabel("Time")
ax3[0].set_ylabel("Values")
ax3[0].legend()
ax3[0].grid()

ax3[1].plot(time, F3_error, label="F3 Relative Error", color="green", linewidth=2)
ax3[1].set_title("F3: Relative Error")
ax3[1].set_xlabel("Time")
ax3[1].set_ylabel("Relative Error")
ax3[1].legend()
ax3[1].grid()

# Adjust layouts and show plots
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()

plt.show()