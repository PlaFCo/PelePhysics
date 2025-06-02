#ifndef GPU_MISC_H
#define GPU_MISC_H

#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <cmath>

#include <PelePhysics.H>


amrex::Real efield_func(
  const amrex::Real x,
  const amrex::Real y,
  const amrex::Real z)
{
  // Example electric field function
  // This is a placeholder; replace with the actual field calculation
  return std::cos(x)*std::cos(y);
}

amrex::Real efield_to_tempe(const amrex::Real efield)
{
  // Convert electric field to electron temperature
  // This is a placeholder; replace with the actual conversion
  return efield * 1e3; // Example conversion factor
}

amrex::Real tempe_to_efield(const amrex::Real tempe)
{
  // Convert electric field to electron temperature
  // This is a placeholder; replace with the actual conversion
  amrex::Real factor = 1.0/efield_to_tempe(1.0); // If linear
  return factor * tempe; // Example conversion factor
}

amrex::Real efield_to_hsrc(
  const amrex::Real efield,
  const amrex::Real rhoYe)
{
  // Convert electric field to energy source term
  // This is a placeholder; replace with the actual conversion
  const amrex::Real factor = 1e-6; // Example conversion factor
  return  rhoYe * efield*efield * factor;
}

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
  X[N2_ID] = 0.9;
  X[N2p_ID] = 0.05;
  X[E_ID] = 0.05;

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


AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
initialize_solution2t(
  int i,
  int j,
  int k,
  int energy_type,
  amrex::Array4<amrex::Real> const& rhoY,
  amrex::Array4<amrex::Real> const& frcExt,
  amrex::Array4<amrex::Real> const& rhoE,
  amrex::Array4<amrex::Real> const& frcEExt,
  amrex::GeometryData const& geomdata,
  amrex::Real temp,
  amrex::Real tempe) noexcept
{
  const amrex::Real* plo = geomdata.ProbLo();
  const amrex::Real* phi = geomdata.ProbHi();
  const amrex::Real* dx = geomdata.CellSize();

  amrex::Real x = plo[0] + (phi[0] - plo[0]) * (i + 0.5) / dx[0];
  amrex::Real y = plo[1] + (phi[1] - plo[1]) * (j + 0.5) / dx[1];
  amrex::Real z = plo[2] + (phi[2] - plo[2]) * (k + 0.5) / dx[2];

  amrex::Real pressure = 0.1 * 1013250.0;
  amrex::Real density = 0.0;
  amrex::Real energy = 0.0;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> X;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> Y;

  // Y mass fractions
  for (int n = 0; n < NUM_SPECIES; n++) {
    X[n] = 0.0;
  }
  X[N2_ID] = 0.9;
  X[N2p_ID] = 0.05;
  X[E_ID] = 0.05;

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
  rhoE(i, j, k, 0) = energy * density;

  amrex::Real efield_value = efield_func(x, y, z);
  tempe = efield_to_tempe(efield_value);
  amrex::Real joule_source = efield_to_hsrc(efield_value, density * Y[E_ID]);
  
  rhoE(i, j, k, 1) = tempe;
  frcEExt(i, j, k, 0) = joule_source;
  frcEExt(i, j, k, 1) = 0.0;
}


#endif