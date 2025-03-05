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
    Tnp1 = temperature
    for temp_iter in range(0, 50):
        mechanism.TPX = Tnp1, 0.1 * ct.one_atm, "N2:0.5,O2:0.5"
        mechanism.transport_model = "mixture-averaged"
        # r = ct.IdealGasConstPressureReactor(mechanism)
        r = ct.IdealGasReactor(mechanism)
        sim = ct.ReactorNet([r])
        time = 0.0
        states = ct.SolutionArray(mechanism, extra=["t"])
        dt = 100
        ndt = 1
        time = dt/ndt
        sim.advance(time)
        Tnp1 = r.T


        if (temp_iter >0):
            f_Tn = Tnp1 - temperature
            Tnp1 = Tn - f_Tn*(Tn-Tnm1)/(f_Tn-f_Tnm1)
            Tnm1 = Tn
            Tn = Tnp1
            f_Tnm1 = f_Tn
        else:
            Tnm1 = temperature
            f_Tn = Tnp1 - temperature
            Tn = Tnp1 
            Tnp1 = Tn - 0.5*f_Tn
            f_Tnm1 = f_Tn

        if (abs(Tn-temperature) < 1.0e-6):
            break


    # print(f"Tn->{Tnp1}  T={r.T}")
    # mechanism.TPX = Tnp1, 0.1 * ct.one_atm, "N2:0.5,O2:0.5"
    # mechanism.transport_model = "Mix"
    # # r = ct.IdealGasConstPressureReactor(mechanism)
    # r = ct.IdealGasReactor(mechanism)
    # sim = ct.ReactorNet([r])
    # time = 0.0
    # states = ct.SolutionArray(mechanism, extra=["t"])
    # dt = 10
    # ndt = 1000

    # mechanism()
    results = numpy.append(mechanism.X,mechanism.T)
    # print(results)
    with open(local_dir + "/cantera_simulation.txt", 'w') as myfile:
        for variable in results:
            myfile.write(str(variable) + ' ')
        myfile.write(str(Tn) + ' ')
        
    # lst = [
    #     {
    #         "time": time,
    #         "temperature": mechanism.T,
    #         "density": mechanism.density,
    #         "viscosity": mechanism.viscosity,
    #     }
    # ]
    # for _n in range(ndt):
    #     time += dt / ndt
    #     sim.advance(time)
    #     states.append(r.thermo.state, t=time * 1e3)
    #     lst.append(
    #         {
    #             "time": time,
    #             "temperature": r.T,
    #             "density": states[-1].density,
    #             "viscosity": states[-1].viscosity,
    #         }
    #     )

    # df = pd.DataFrame(lst)

    # # Write result to file
    # oname = "results.txt"
    # df.to_csv(oname, index=False)


if __name__ == "__main__":
    main()
