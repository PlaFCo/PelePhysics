#include <iostream>
#include <vector>

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>

#ifdef AMREX_USE_GPU
#include <AMReX_SUNMemory.H>
#endif

#include "mechanism.H"
#include <initialize.H>

#include <PelePhysics.H>
#include <ReactorBase.H>

#include "utils/initFunctions.H"
#include "utils/reactFunctions.H"

int
main(int argc, char* argv[])
{
  amrex::Initialize(argc, argv);

#ifdef AMREX_USE_GPU
  amrex::sundials::Initialize();
#endif
  {

    amrex::Real strt_time = amrex::ParallelDescriptor::second();
    BL_PROFILE_VAR("main::main()", pmain);

    // Init: Read input, initialize transport, geom, data
    // Parse the relevant inputs
    std::string chem_integrator;
    std::string pltfile;
    std::string chkfile, reactFormat;
    bool do_plt;
    int initFromChk, reactFunc, ode_ncells, ndt, ode_iE, use_typ_vals,
      max_grid_size;
    amrex::Real dt, rtol, atol, temperature;
    std::array<int, 3> ncells;
    amrex::ParmParse pp;
    amrex::ParmParse ppode("ode");
    amrex::ParmParse ppstate("state");
    parse_input(
      pp, ppode, ppstate, chem_integrator, do_plt, pltfile, initFromChk,
      chkfile, reactFormat, reactFunc, ode_ncells, dt, ndt, ode_iE, rtol, atol,
      use_typ_vals, ncells, max_grid_size, temperature);

    // Initialize transport
    pele::physics::PeleParams<pele::physics::transport::TransParm<
      pele::physics::PhysicsType::eos_type,
      pele::physics::PhysicsType::transport_type>>
      trans_parms;
    trans_parms.initialize();

    // Initialize reactor object inside OMP region, including tolerances
    BL_PROFILE_VAR("main::reactor_info()", reactInfo);
    std::unique_ptr<pele::physics::reactions::ReactorBase> reactor =
      pele::physics::reactions::ReactorBase::create(chem_integrator);
    reactor->init(ode_iE, ode_ncells);
    BL_PROFILE_VAR_STOP(reactInfo);

    // Initialize Geometry
    int finest_level = 0;
    amrex::Vector<amrex::Geometry> geoms;
    amrex::Vector<amrex::BoxArray> grids;
    amrex::Vector<amrex::DistributionMapping> dmaps;
    BL_PROFILE_VAR("main::geometry_setup", GeomSetup);
    initialize_geom(
      geoms, grids, dmaps, finest_level, ncells, ndt, dt, max_grid_size);
    BL_PROFILE_VAR_STOP(GeomSetup);

    // Initialize Data
    BL_PROFILE_VAR("main::initialize_solution()", InitData);
    int num_grow = 0;
    amrex::Vector<amrex::MultiFab> mf(finest_level + 1);
    amrex::Vector<amrex::MultiFab> rY_source_ext(finest_level + 1);
    amrex::Vector<amrex::MultiFab> mfE(finest_level + 1);
    amrex::Vector<amrex::MultiFab> rY_source_energy_ext(finest_level + 1);
    amrex::Vector<amrex::MultiFab> fctCount(finest_level + 1);
    amrex::Vector<amrex::iMultiFab> dummyMask(finest_level + 1);
    initialize_data(
      num_grow, mf, rY_source_ext, mfE, rY_source_energy_ext, fctCount,
      dummyMask, finest_level, geoms, grids, dmaps, ode_iE, temperature)
    BL_PROFILE_VAR_STOP(InitData);

    // React
    amrex::Print() << " \n STARTING THE ADVANCE \n";

    amrex::Real f_Tn;
    amrex::Real f_Tnm1;
    amrex::Real Tn;
    amrex::Real Tnm1;
    amrex::Real Tnp1 = temperature + 500;
    for( int temp_iter = 0; temp_iter < 50; temp_iter++) {

      for (int lev = 0; lev <= finest_level; ++lev) {
        amrex::Real lvl_strt = amrex::ParallelDescriptor::second();
        BL_PROFILE_VAR("Advance_Level" + std::to_string(lev), Advance);
#ifdef AMREX_USE_OMP
        const auto tiling = amrex::MFItInfo().SetDynamic(true);
#pragma omp parallel
#else
        const bool tiling = amrex::TilingIfNotGPU();
#endif
      for (amrex::MFIter mfi(mf[lev], tiling); mfi.isValid(); ++mfi) {

          int omp_thread = 0;
#ifdef AMREX_USE_OMP
          omp_thread = omp_get_thread_num();
#endif
          // Reaction at constant volume
          if (reactFunc == 1) {
            integrate_isochoric(
              lev, dt, ndt, omp_thread, mfi, mf, rY_source_ext, mfE,
              rY_source_energy_ext, fctCount, dummyMask, reactor, trans_parms);
            Tnp1 = mf[lev].array(mfi, NUM_SPECIES)(0, 0, 0);
          } else if (reactFunc == 2) {
            integrate_isobaric(
              lev, dt, ndt, omp_thread, ode_ncells, mfi, mf, rY_source_ext, mfE,
              rY_source_energy_ext, fctCount, dummyMask, reactor);
            Tnp1 = mf[lev].array(mfi, NUM_SPECIES)(0, 0, 0);
          }
        }
        BL_PROFILE_VAR_STOP(Advance);
        amrex::Real lvl_run_time = amrex::ParallelDescriptor::second() - lvl_strt;
        amrex::ParallelDescriptor::ReduceRealMax(
          lvl_run_time, amrex::ParallelDescriptor::IOProcessorNumber());

      }
      if (temp_iter>0){
        f_Tn = Tnp1 - temperature;
        Tnp1 = Tn - f_Tn*(Tn-Tnm1)/(f_Tn-f_Tnm1); 
        Tnm1 = Tn;
        Tn = Tnp1;
        f_Tnm1 = f_Tn;
      }
      else{
        Tnm1 = temperature;
        f_Tn = Tnp1-temperature;
        Tn = Tnp1;
        Tnp1 = Tn - 0.5* f_Tn;
        f_Tnm1 = f_Tn;
      }
      
      reset_temperature(
        num_grow, mf, rY_source_ext, mfE, rY_source_energy_ext, fctCount,
        dummyMask, finest_level, geoms, grids, dmaps, ode_iE, Tnp1)
      BL_PROFILE_VAR_STOP(InitData);

      if (std::abs(Tn-temperature) < 1.0e-3) {
        amrex::Print() << temp_iter <<": Tn->" << Tn << "  T="<<temperature<<  "\n";
        break;
      }
    }

    // TODO multilevel max.
    {
      amrex::Vector<double> typ_vals(NUM_SPECIES + 1);
      amrex::Print() << "ode.typ_vals= ";
      for (int i = 0; i < NUM_SPECIES + 1; ++i) {
        amrex::Print() << std::max(1.e-10, mf[0].max(i)) << " ";
      }
      amrex::Print() << std::endl;
    }

    // Finalize
    trans_parms.deallocate();
    BL_PROFILE_VAR_STOP(pmain);
    amrex::Real run_time = amrex::ParallelDescriptor::second() - strt_time;
    amrex::ParallelDescriptor::ReduceRealMax(
      run_time, amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::Print() << " \n >> React::main() " << run_time << "\n\n";
  }
#ifdef AMREX_USE_GPU
  amrex::sundials::Finalize();
#endif
  amrex::Finalize();

  return 0;
}
