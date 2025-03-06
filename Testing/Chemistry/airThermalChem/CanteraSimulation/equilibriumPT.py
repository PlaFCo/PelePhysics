"""Find equilibrium composition for a given pressure and temperature."""
import argparse
import pathlib

import cantera as ct
import pandas as pd
import numpy


def main():
    local_dir = str(pathlib.Path(__file__).parent.resolve())
    mechanism_dir = local_dir.split("/Testing/")[0]
    airthermal_dir = mechanism_dir + "/Mechanisms/airThermal"
    chem_name = airthermal_dir + "/mechanism.yaml"
    mechanism = ct.Solution(chem_name)

    temperature = 4000.0
    mechanism.TPX = temperature, 0.1 * ct.one_atm, "N2:0.5,O2:0.5"
    mechanism.equilibrate("TP", solver="gibbs")

    results = numpy.append(mechanism.X,mechanism.T)
    with open(local_dir + "/cantera_equilibrium.txt", 'w') as myfile:
        for variable in results:
            myfile.write(str(variable) + ' ')
        myfile.write(str(0.0) + ' ')

    with open(local_dir + "/species_names.txt", "w") as species_names_file:
        for  name in mechanism.species_names:
            species_names_file.write(name + " ")

if __name__ == "__main__":
    main()
