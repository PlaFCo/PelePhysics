"""Run a homogeneous reactor in Cantera."""
#Iterates over initial temperature to find equilibrium composition
import argparse
import pathlib

import cantera as ct
import pandas as pd
import numpy


def main():
    """Run the reactor."""
    # parser = argparse.ArgumentParser(description="Cantera homogeneous reactor")
    # parser.add_argument("-f", "--fname", help="Mechanism file", type=str, required=True)
    # args = parser.parse_args()
    local_dir = str(pathlib.Path(__file__).parent.resolve())
    mechanism_dir = local_dir.split("/Testing/")[0]
    airthermal_dir = mechanism_dir + "/Mechanisms/airThermal"
    chem_name = airthermal_dir + "/mechanism.yaml"
    mechanism = ct.Solution(chem_name)

    temperature = 4000.0
    mechanism.TPX = temperature, 0.1 * ct.one_atm, "N2:0.5,O2:0.5"
    mechanism.equilibrate("TP", solver="gibbs")
    # mechanism()

    results = numpy.append(mechanism.X,mechanism.T)
    # print(results)
    with open(local_dir + "/cantera_equilibrium.txt", 'w') as myfile:
        for variable in results:
            myfile.write(str(variable) + ' ')
        myfile.write(str(0.0) + ' ')

if __name__ == "__main__":
    main()
