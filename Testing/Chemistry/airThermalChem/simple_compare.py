import subprocess
import pathlib
import numpy
import matplotlib.pyplot as plt


def main():

    local_dir = str(pathlib.Path(__file__).parent.resolve())
    print("Here is the file")

    # Run cantera equilibrium calculation
    cantera_dir = local_dir + "/CanteraSimulation"
    subprocess.run("python " + cantera_dir + "/equilibriumPT.py", shell=True)

    # Run cantera simulation
    cantera_dir = local_dir + "/CanteraSimulation"
    subprocess.run("python " + cantera_dir + "/homogeneous-reactor.py", shell=True)

    # Run pelephysics simulation
    pele_dir = local_dir + "/PelePhysicsSimulation"
    subprocess.run(pele_dir + "/Pele3d.gnu.ex " + pele_dir + "/inputs-cvode-0d.inp", shell=True)    

    # Compare results
    equilibrium_data = numpy.loadtxt(local_dir + "/CanteraSimulation/cantera_equilibrium.txt")
    cantera_data = numpy.loadtxt(local_dir + "/CanteraSimulation/cantera_simulation.txt")
    pele_data = numpy.loadtxt(local_dir + "/PelePhysicsSimulation/pele_simulation.txt")

    plt.plot(equilibrium_data[:-2],label="Equilibrium")
    plt.plot(cantera_data[:-2],label="Cantera")
    plt.plot(pele_data[:-2],label="PelePhysics")
    plt.xlabel("species id")
    plt.ylabel("molar fraction")
    plt.legend()
    plt.savefig(local_dir + "/comparison_X.png",dpi=300)
    plt.clf()

    plt.scatter(1.0,cantera_data[-2],marker='s',label="Cantera final")
    plt.scatter(0.0,cantera_data[-1],marker='s',label="Cantera initial")
    plt.scatter(1.0,pele_data[-2],   marker='v',label="PelePhysics final")
    plt.scatter(0.0,pele_data[-1],   marker='v',label="PelePhysics initial")
    plt.scatter(1.0,equilibrium_data[-2],marker='+',label="Equilibrium final")
    plt.xlabel("a.u.")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.savefig(local_dir + "/comparison_T.png",dpi=300)
    plt.clf()

if __name__ == "__main__":
    main()