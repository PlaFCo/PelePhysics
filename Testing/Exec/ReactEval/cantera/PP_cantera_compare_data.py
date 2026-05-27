import numpy as np
import matplotlib.pyplot as plt

# File names
file1 = "reactor_output.txt"
file2 = "../results/Ereact_GRI.txt"

# Load data (assumes whitespace-separated text)
data1 = np.loadtxt(file1)
data2 = np.loadtxt(file2)

# header
with open(file1, "r") as f:
    for line in f:
        if line.startswith("#"):
            header = line.strip("#").strip()
            break
labels = header.split()

# Extract time and variables
data1 = np.loadtxt(file1, comments="#")
t1 = data1[:, 0]
data2 = np.loadtxt(file2, comments="#")
t2 = data2[:, 0]

# Create figure with 6 subplots
fig, axes = plt.subplots(6, 1, sharex=True, figsize=(8, 12))

for i in range(6):
    col_label = labels[i+1] if len(labels) > i+1 else f"Col {i+1}"

    axes[i].plot(t1, data1[:, i+1], label="CANTERA")
    #axes[i].plot(t1, data1[:, i+1], label=f"Cantera ({col_label})")
    axes[i].plot(t2, data2[:, i+1], linestyle="--", label="PelePhysics")

    #axes[i].set_ylabel(f"Col {i+1}")
    axes[i].set_ylabel(col_label)
    axes[i].grid()

# Label x-axis only on bottom plot
axes[-1].set_xlabel("Time [s]")

# Add legend to first plot (or all if you prefer)
axes[0].legend()

plt.tight_layout()
#plt.show()
plt.savefig("PPvsCantera.png", dpi=300)
plt.close()
