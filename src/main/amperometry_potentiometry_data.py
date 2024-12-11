import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

dataset_location = (
    r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
    r"\amperometry_potentiometry"
)

file_path1 = 'Chronopotentiometry.csv'  # Replace 'your_file.csv' with the actual file path
file_path2 = 'Chronoamperometry.csv'  # Replace 'your_file.csv' with the actual file path
data1 = pd.read_csv(os.path.join(dataset_location, file_path1))
data2 = pd.read_csv(os.path.join(dataset_location, file_path2))

# Extract data from two columns
column11 = data1['Elapsed Time (s)']  # Replace 'column1_name' with the actual name of the first column
column12 = data1['Potential (V)']  # Replace 'column2_name' with the actual name of the second column
column21 = data2['Elapsed Time (s)']  # Replace 'column1_name' with the actual name of the first column
column22 = data2['Current (A)']*1000/56  # Replace 'column2_name' with the actual name of the second column

# Create a figure and two subplots with a 2:1 aspect ratio
fig, axs = plt.subplots(2, 1)

# Plot the data in the first subplot
axs[0].plot(column11, column12, linestyle='-')
axs[0].set_title('Chronopotentiogram: Voltage at current density = 2 mA/cm$^2$')
axs[0].set_xlabel('Time (seconds)')
axs[0].set_ylabel('Potential (V)')

# Plot the data in the second subplot (you can customize this plot as needed)
axs[1].plot(column21, column22, linestyle='-',color='orange')  # Example of a different plot
axs[1].set_title('Chronoamperogram: Current at potential = 2.5 V')  # Title for the second subplot
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Current density (mA/cm$^2$)')

plt.tight_layout()  # Adjust subplots to prevent overlap
plt.savefig(f"{dataset_location}/chronopotentiometry_chronoamperometry.pdf", format='pdf')
plt.show()

R = np.linspace(1, 300, 300)
plt.figure()
V_values = range(9, 5, -3)
j_curves = []
for V in V_values:
    j = (V * 1000) / (56 * (R + 42.85))
    j_curves.append(j)
    plt.plot(R, j)

plt.fill_between(R, j_curves[0], j_curves[1], color='gray', alpha=0.5)
plt.xlabel('R')
plt.ylabel('j')
plt.tight_layout()  # Adjust subplots to prevent overlap
plt.savefig(f"{dataset_location}/j_curve.pdf", format='pdf')
plt.show()