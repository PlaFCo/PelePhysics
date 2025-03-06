import subprocess
import pathlib
import numpy
import matplotlib.pyplot as plt

def main():

    local_dir = str(pathlib.Path(__file__).parent.resolve())
    print("Here is the file")

    # run cantera equilibrium calculation
    cantera_dir = local_dir + "/CanteraSimulation"
    subprocess.run("python " + cantera_dir + "/equilibriumPT.py", shell=True)

    # run cantera simulation
    cantera_dir = local_dir + "/CanteraSimulation"
    subprocess.run("python " + cantera_dir + "/homogeneous-reactor.py", shell=True)

    # run pelephysics simulation
    pele_dir = local_dir + "/PelePhysicsSimulation"
    subprocess.run(pele_dir + "/Pele3d.gnu.ex " + pele_dir + "/inputs-cvode-0d.inp", shell=True)    

    # read species names
    species_names = numpy.loadtxt(local_dir + "/CanteraSimulation/species_names.txt",dtype=str)

    # compare results
    equilibrium_data = numpy.loadtxt(local_dir + "/CanteraSimulation/cantera_equilibrium.txt")
    cantera_data = numpy.loadtxt(local_dir + "/CanteraSimulation/cantera_simulation.txt")
    pele_data = numpy.loadtxt(local_dir + "/PelePhysicsSimulation/pele_simulation.txt")

    # sanity checks
    print("Sum(X[:]) equilibrium->",numpy.sum(equilibrium_data[:-2]))
    print("Sum(X[:]) cantera->",numpy.sum(cantera_data[:-2]))
    print("Sum(X[:]) pele->",numpy.sum(pele_data[:-2]))

    # plot molar fractions
    plt.plot(cantera_data[:-2],marker='s',label="Cantera")
    plt.plot(pele_data[:-2]/numpy.sum(pele_data[:-2]),marker='v',label="PelePhysics")
    plt.plot(equilibrium_data[:-2],marker='+',label="Equilibrium")
    plt.xlabel("species id")
    plt.xticks(range(len(species_names)),species_names,rotation=90)
    plt.ylabel("molar fraction")
    plt.legend()
    plt.savefig(local_dir + "/comparison_X.png",dpi=300)
    plt.yscale("log")
    plt.savefig(local_dir + "/comparison_X_ylog.png",dpi=300)
    plt.clf()

    # plot final and initial temperatures
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