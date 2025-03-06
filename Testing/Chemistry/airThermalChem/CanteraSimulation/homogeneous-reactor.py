"""
Run a homogeneous reactor in Cantera.
Iterates over initial temperature to find equilibrium composition for a given pressure and temperature
"""
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
    Tnp1 = temperature
    for temp_iter in range(0, 50):
        mechanism.TPX = Tnp1, 0.1 * ct.one_atm, "N2:0.5,O2:0.5"
        mechanism.transport_model = "mixture-averaged"
        r = ct.IdealGasConstPressureReactor(mechanism)
        sim = ct.ReactorNet([r])
        time = 0.0
        states = ct.SolutionArray(mechanism, extra=["t"])
        dt = 1000
        ndt = 100
        time = dt/ndt
        sim.advance(time)
        Tnp1 = r.T

        if (temp_iter >0): # Newton
            f_Tn = Tnp1 - temperature
            Tnp1 = Tn - f_Tn*(Tn-Tnm1)/(f_Tn-f_Tnm1)
            Tnm1 = Tn
            Tn = Tnp1
            f_Tnm1 = f_Tn
        else: # secant
            Tnm1 = temperature
            f_Tn = Tnp1 - temperature
            Tn = Tnp1 
            Tnp1 = temperature + 0.5*f_Tn
            f_Tnm1 = f_Tn

        if (abs(Tn-Tnm1) < 1.0e-6):
            break

    results = numpy.append(mechanism.X,mechanism.T)
    with open(local_dir + "/cantera_simulation.txt", 'w') as myfile:
        for variable in results:
            myfile.write(str(variable) + ' ')
        myfile.write(str(Tn) + ' ')

if __name__ == "__main__":
    main()
