"""Run a homogeneous reactor in Cantera."""
#Iterates over initial temperature to find equilibrium composition
import argparse

import cantera as ct
import pandas as pd


def main():
    """Run the reactor."""
    # parser = argparse.ArgumentParser(description="Cantera homogeneous reactor")
    # parser.add_argument("-f", "--fname", help="Mechanism file", type=str, required=True)
    # args = parser.parse_args()

    chem_name = "../../../../Mechanisms/airThermal/mechanism.yaml"
    mechanism = ct.Solution(chem_name)

    temperature = 4000.0
    mechanism.TPX = temperature, 0.1 * ct.one_atm, "N2:0.5,O2:0.5"
    mechanism.equilibrate("TP", solver="gibbs")
    mechanism()


if __name__ == "__main__":
    main()
