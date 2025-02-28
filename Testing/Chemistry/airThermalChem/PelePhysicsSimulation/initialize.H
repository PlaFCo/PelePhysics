#ifndef GPU_MISC_H
#define GPU_MISC_H

#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <cmath>

#include <PelePhysics.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
initialize_solution(
  int i,
  int j,
  int k,
  int energy_type,
  amrex::Array4<amrex::Real> const& rhoY,
  amrex::Array4<amrex::Real> const& frcExt,
  amrex::Array4<amrex::Real> const& rhoE,
  amrex::Array4<amrex::Real> const& frcEExt,
  amrex::GeometryData const& geomdata,
  amrex::Real temp) noexcept
{
  // const amrex::Real* plo = geomdata.ProbLo();
  // const amrex::Real* phi = geomdata.ProbHi();
  // const amrex::Real* dx = geomdata.CellSize();

  amrex::Real pressure = 0.1 * 1013250.0;
  amrex::Real density = 0.0;
  amrex::Real energy = 0.0;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> X;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> Y;

  // Y mass fractions
  for (int n = 0; n < NUM_SPECIES; n++) {
    X[n] = 0.0;
  }
  X[O2_ID] = 0.5;
  X[N2_ID] = 0.5;

  auto eos = pele::physics::PhysicsType::eos();
  eos.X2Y(&X[0], &Y[0]);

  // Temperature
  if (energy_type == 1) {
    // get rho and E
    eos.PYT2RE(pressure, &Y[0], temp, density, energy);
  } else {
    // get rho and H
    eos.PYT2R(pressure, &Y[0], temp, density);
    eos.TY2H(temp, &Y[0], energy);
  }
  for (int n = 0; n < NUM_SPECIES; n++) {
    rhoY(i, j, k, n) = Y[n] * density;
    frcExt(i, j, k, n) = 0.0;
  }
  rhoY(i, j, k, NUM_SPECIES) = temp;
  rhoE(i, j, k) = energy * density;
  frcEExt(i, j, k) = 0.0;
}

#endif
