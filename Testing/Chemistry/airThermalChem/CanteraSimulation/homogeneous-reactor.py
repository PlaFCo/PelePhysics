"""Run a homogeneous reactor in Cantera."""
import argparse

import cantera as ct
import pandas as pd


def main():
    """Run the reactor."""
    parser = argparse.ArgumentParser(description="Cantera homogeneous reactor")
    parser.add_argument("-f", "--fname", help="Mechanism file", type=str, required=True)
    args = parser.parse_args()

    chem_name = "../../../Mechanisms/airThermal/mechanism.yaml"
    mechanism = ct.Solution(chem_name)
    mechanism.TPX = 3000.0, 0.1 * ct.one_atm, "N2:0.8,O2:0.2"
    mechanism.transport_model = "Mix"
    # r = ct.IdealGasConstPressureReactor(mechanism)
    r = ct.IdealGasReactor(mechanism)
    sim = ct.ReactorNet([r])
    time = 0.0
    states = ct.SolutionArray(mechanism, extra=["t"])
    dt = 10
    ndt = 1000

    mechanism()

    lst = [
        {
            "time": time,
            "temperature": mechanism.T,
            "density": mechanism.density,
            "viscosity": mechanism.viscosity,
        }
    ]
    for _n in range(ndt):
        time += dt / ndt
        sim.advance(time)
        states.append(r.thermo.state, t=time * 1e3)
        lst.append(
            {
                "time": time,
                "temperature": r.T,
                "density": states[-1].density,
                "viscosity": states[-1].viscosity,
            }
        )

    df = pd.DataFrame(lst)

    # Write result to file
    oname = "results.txt"
    df.to_csv(oname, index=False)


if __name__ == "__main__":
    main()
