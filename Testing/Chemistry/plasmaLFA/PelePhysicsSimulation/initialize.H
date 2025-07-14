#ifndef GPU_MISC_H
#define GPU_MISC_H

#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <cmath>

#include <PelePhysics.H>


amrex::Real redEfield_func(
  const amrex::Real x,
  const amrex::Real y,
  const amrex::Real z)
{  // Example electric field function
   // [redEfield] = Td
  return 1.1;
}

// Nitrogen
amrex::Real redEfield_to_tempe(const amrex::Real redEfield) // units: Td
{
  // Convert electric field to electron temperature
  // [redEfield] = Td
  // [tempe] = eV
  amrex::Real redEfield2 = redEfield*redEfield;
  amrex::Real redEfield3 = redEfield2 * redEfield;
  amrex::Real redEfield4 = redEfield3 * redEfield;
  amrex::Real redEfield5 = redEfield4 * redEfield;
  amrex::Real redEfield6 = redEfield5 * redEfield;
  amrex::Real redEfield7 = redEfield6 * redEfield;
  amrex::Real coeff[8] = {2.80762643e-09, -3.44166229e-07, 1.72107619e-05, -4.52557319e-04,
  6.73930552e-03, -5.73558598e-02, 2.73334591e-01, 1.44603888e-02};
  amrex::Real tempe = coeff[0] + coeff[1]*redEfield + coeff[2]*redEfield2 + coeff[3]*redEfield3
    + coeff[4]*redEfield4 + coeff[5]*redEfield5 + coeff[6]*redEfield6 + coeff[7]*redEfield7;
  return tempe; // Example conversion factor
}

amrex::Real tempe_to_redEfield(const amrex::Real tempe) 
{
  // Convert electric field to electron temperature
  // [tempe] = eV
  // [redEfield] = Td
  amrex::Real tempe2 = tempe*tempe;
  amrex::Real tempe3 = tempe2 * tempe;
  amrex::Real tempe4 = tempe3 * tempe;
  amrex::Real tempe5 = tempe4 * tempe;
  amrex::Real tempe6 = tempe5 * tempe;
  amrex::Real tempe7 = tempe6 * tempe;
  amrex::Real coeff[8] = { 2.50105883e+04, -5.44366566e+04,  4.71853693e+04, -2.05715135e+04,
  4.73687754e+03, -5.51369499e+02, 3.27139044e+01, -5.39064133e-01};
  return coeff[0] + coeff[1]*tempe + coeff[2]*tempe2 + coeff[3]*tempe3
    + coeff[4]*tempe4 + coeff[5]*tempe5 + coeff[6]*tempe6 + coeff[7]*tempe7;
}

amrex::Real electron_mobilityN2(
  amrex::Real redEfield)
  {
    // [redEfield] = Td
    // [mobility] = m-1 s-1 V-1
    amrex::Real log_redEfield = std::log10(redEfield);
    amrex::Real log_redEfield2 = log_redEfield * log_redEfield;
    amrex::Real log_redEfield3 = log_redEfield2 * log_redEfield;
    amrex::Real log_redEfield4 = log_redEfield3 * log_redEfield;
    amrex::Real log_redEfield5 = log_redEfield4 * log_redEfield;
    amrex::Real log_redEfield6 = log_redEfield5 * log_redEfield;
    amrex::Real log_redEfield7 = log_redEfield6 * log_redEfield;
    amrex::Real coeff[8] = {-5.07985187e+23,  3.76472668e+22,  4.04491067e+24, -1.87198611e+24,
 -9.29756879e+24,  1.04424317e+25, -5.18591049e+24,  4.08807283e+24};
    return coeff[0] + coeff[1]*log_redEfield + coeff[2]*log_redEfield2 + coeff[3]*log_redEfield3
      + coeff[4]*log_redEfield4 + coeff[5]*log_redEfield5 + coeff[6]*log_redEfield6 + coeff[7]*log_redEfield7;
}

amrex::Real redEfield_to_hsrc(
  const amrex::Real redEfield,
  const amrex::Real rhoYe,
  const amrex::Real N)
{
  // Convert electric field to energy/enthalpy source term
  // [redEfield] = Td
  // [rhoYe] = kg m^-3
  // [N] = m^-3
  // [hsrc] = J m^-3 s^-1
  // m-3 * V m^2 * V * m^2 * m-1 s-1 V-1 * e  * m-3
  // m-3 s-1
  amrex::Real ne = rhoYe / 9.109e-31; // Convert mass density to number density
  amrex::Print() << ne << "\n";
  const amrex::Real factor = 1.0e-21*1.0e-21; // Td to V m^2
  const amrex::Real C = 1.602e-19; // Elementary charge in Coulombs
  amrex::Real redMob = electron_mobilityN2(redEfield);
  return  std::abs(  ne * factor*redEfield*N*redEfield*N * redMob/N * C ); // J m^-3 s^-1
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
  const amrex::Real temp[NUM_TEMP]) noexcept
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
  rhoY(i, j, k, NUM_SPECIES) = temp[0];
#ifdef PELE_USE_NLTE
  rhoY(i, j, k, NUM_SPECIES + 1) = temp[1];
#endif
  rhoE(i, j, k) = energy * density;
  frcEExt(i, j, k) = 0.0;
}


// AMREX_GPU_HOST_DEVICE
// AMREX_FORCE_INLINE
// void
// initialize_solution2t(
//   int i,
//   int j,
//   int k,
//   int energy_type,
//   amrex::Array4<amrex::Real> const& rhoY,
//   amrex::Array4<amrex::Real> const& frcExt,
//   amrex::Array4<amrex::Real> const& rhoE,
//   amrex::Array4<amrex::Real> const& frcEExt,
//   amrex::GeometryData const& geomdata,
//   amrex::Real temp,
//   amrex::Real tempe) noexcept
// {
//   const amrex::Real* plo = geomdata.ProbLo();
//   const amrex::Real* phi = geomdata.ProbHi();
//   const amrex::Real* dx = geomdata.CellSize();

//   amrex::Real x = plo[0] + (phi[0] - plo[0]) * (i + 0.5) / dx[0];
//   amrex::Real y = plo[1] + (phi[1] - plo[1]) * (j + 0.5) / dx[1];
//   amrex::Real z = plo[2] + (phi[2] - plo[2]) * (k + 0.5) / dx[2];

//   amrex::Real pressure = 0.1 * 1013250.0;
//   amrex::Real density = 0.0;
//   amrex::Real energy = 0.0;
//   amrex::GpuArray<amrex::Real, NUM_SPECIES> X;
//   amrex::GpuArray<amrex::Real, NUM_SPECIES> Y;

//   // Y mass fractions
//   for (int n = 0; n < NUM_SPECIES; n++) {
//     X[n] = 0.0;
//   }
//   X[N2_ID] = 0.9;
//   X[N2p_ID] = 0.05;
//   X[E_ID] = 0.05;

//   auto eos = pele::physics::PhysicsType::eos();
//   eos.X2Y(&X[0], &Y[0]);

//   // Temperature
//   if (energy_type == 1) {
//     // get rho and E
//     eos.PYT2RE(pressure, &Y[0], temp, density, energy);
//   } else {
//     // get rho and H
//     eos.PYT2R(pressure, &Y[0], temp, density);
//     eos.TY2H(temp, &Y[0], energy);
//   }
//   for (int n = 0; n < NUM_SPECIES; n++) {
//     rhoY(i, j, k, n) = Y[n] * density;
//     frcExt(i, j, k, n) = 0.0;
//   }
//   rhoY(i, j, k, NUM_SPECIES) = temp;
//   rhoE(i, j, k, 0) = energy * density;

//   amrex::Real redEfield_value = redEfield_func(x, y, z);
//   tempe = redEfield_to_tempe(redEfield_value);
//   amrex::Real joule_source = redEfield_to_hsrc(redEfield_value, density * Y[E_ID]);
  
//   rhoE(i, j, k, 1) = tempe;
//   frcEExt(i, j, k, 0) = joule_source;
//   frcEExt(i, j, k, 1) = 0.0;
// }


#endif