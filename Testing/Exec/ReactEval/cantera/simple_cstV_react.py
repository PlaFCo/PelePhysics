import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# Gas object
gas = ct.Solution("/Users/afelden/Documents/CODES/PeleLMeX_PlaFCo/Submodules/PelePhysics/Mechanisms/grimech30/mechanism.yaml")

# Initial conditions: pure methane
gas.TPX = 1800.0, ct.one_atm, "CH4:1.0"

# Reactor
reactor = ct.IdealGasReactor(gas)
sim = ct.ReactorNet([reactor])

# species of interest
species_names = ["CH4", "H2", "C2H2", "C2H4", "C2H6"]

# Integration setup
t_end = 1.0
time = 0.0

# Storage arrays
times = []
T_hist = []
rhoY_hist = {sp: [] for sp in species_names}

# Open file and write header
with open("reactor_output.txt", "w") as f:
    #Header    
    #f.write("# time [s], Temperature [K], rho*Y_CH4 [kg/m^3]\n")
    header = "# time[s] T[K] " + " ".join([f"rhoY_{sp}[kg/m3]" for sp in species_names]) + "\n"
    f.write(header)

    # Integrate
    while time < t_end:
        time = sim.step()
    
        rho = reactor.thermo.density
        # Compute rhoY for each species        
        rhoY_values = []        
        for sp in species_names:
            Y = reactor.thermo[sp].Y[0]
            val = rho * Y
            rhoY_hist[sp].append(val)
            rhoY_values.append(val)
    
        times.append(time)
        T_hist.append(reactor.T)
    
        #Write to file
        f.write(
            f"{time:.6e} {reactor.T:.6e} " +
            " ".join([f"{v:.6e}" for v in rhoY_values]) +
            "\n"
        )

# Plot
fig, ax = plt.subplots(4, 1, sharex=True)

# Temperature plot
ax[0].plot(times, T_hist)
ax[0].set_ylabel("Temperature [K]")
ax[0].grid()

# rho*Y_CH4 plot
ax[1].plot(times, rhoY_hist["CH4"])
ax[1].set_ylabel(r"$\rho Y_{CH4}$ [kg/m^3]")
ax[1].set_xlabel("Time [s]")
ax[1].grid()

# rho*Y_C2H2 plot
ax[2].plot(times, rhoY_hist["C2H2"])
ax[2].set_ylabel(r"$\rho Y_{C2H2}$ [kg/m^3]")
ax[2].set_xlabel("Time [s]")
ax[2].grid()

# rho*Y_H2 plot
ax[3].plot(times, rhoY_hist["H2"])
ax[3].set_ylabel(r"$\rho Y_{H2}$ [kg/m^3]")
ax[3].set_xlabel("Time [s]")
ax[3].grid()

plt.tight_layout()
plt.show()

