#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/* Elements
0  N
*/

// Species
#define N2_ID 0
#define N_ID 1

#define NUM_GAS_ELEMENTS 1  // Elements in the homogeneous phase
#define NUM_GAS_SPECIES 2   // Species in the homogeneous phase
#define NUM_GAS_REACTIONS 1 // Reactions in the homogeneous phase

#define SITE_DENSITY 0.000000E+00 // mol/cm^2

#define NUM_SURFACE_ELEMENTS 0  // Additional elements in heterogeneous phase
#define NUM_SURFACE_SPECIES 0   // Species in the heterogeneous phase
#define NUM_SURFACE_REACTIONS 0 // Reactions in the heterogeneous phase

#define NUM_ELEMENTS (NUM_GAS_ELEMENTS + NUM_SURFACE_ELEMENTS)
#define NUM_SPECIES (NUM_GAS_SPECIES + NUM_SURFACE_SPECIES)
#define NUM_REACTIONS (NUM_GAS_REACTIONS + NUM_SURFACE_REACTIONS)

#define NUM_IONS 0

#define NUM_FIT 4

//  ALWAYS on CPU stuff -- can have different def depending on if we are CPU or
//  GPU based. Defined in mechanism.cpp
void atomicWeight(amrex::Real* awt);
//  MISC
void CKAWT(amrex::Real* awt);
void CKNCF(int* ncf);
void CKSYME_STR(amrex::Vector<std::string>& ename);
void CKSYMS_STR(amrex::Vector<std::string>& kname);
void GET_RMAP(int* _rmap);
void CKINU(const int i, int& nspec, int* ki, int* nu);
void CKKFKR(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real* x,
  amrex::Real* q_f,
  amrex::Real* q_r);
void progressRateFR(
  amrex::Real* q_f, amrex::Real* q_r, amrex::Real* sc, amrex::Real T);
//  SPARSE INFORMATION
void SPARSITY_INFO(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST_SIMPLIFIED(int* nJdata, const int* consP);
void
SPARSITY_PREPROC_CSC(int* rowVals, int* colPtrs, const int* consP, int NCELLS);
void SPARSITY_PREPROC_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(
  int* rowVals, int* colPtrs, int* indx, const int* consP);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(
  int* colVals, int* rowPtr, const int* consP, int base);

// A few mechanism parameters
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKINDX(int& mm, int& kk, int& ii, int& nfit)
{
  mm = 1;
  kk = 2;
  ii = 1;
  nfit = -1; // Why do you need this anyway ?
}

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[2] = {
  0.0356964374955379, // N2
  0.0713928749910759, // N
};
#endif
const amrex::Real h_global_imw[2] = {
  0.0356964374955379, // N2
  0.0713928749910759, // N
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[2] = {
  28.014000, // N2
  14.007000, // N
};
#endif
const amrex::Real h_global_mw[2] = {
  28.014000, // N2
  14.007000, // N
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_imw(amrex::Real* imw_new)
{
  imw_new[0] = 0.0356964374955379; // N2
  imw_new[1] = 0.0713928749910759; // N
}

//  inverse molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
imw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_imw[n];
#else
  return h_global_imw[n];
#endif
}
//  molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_mw(amrex::Real* mw_new)
{
  mw_new[0] = 28.014000; // N2
  mw_new[1] = 14.007000; // N
}

//  molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
mw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_mw[n];
#else
  return h_global_mw[n];
#endif
}

//  Returns R, Rc, Patm
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRP(amrex::Real& ru, amrex::Real& ruc, amrex::Real& pa)
{
  ru = 8.31446261815324e+07;
  ruc = 1.98721558317399615845;
  pa = 1.01325e+06;
}

// compute Cv/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cv_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = +2.21037150e+04 * invT2 - 3.81846182e+02 * invT +
                 5.08273836e+00 - 8.53091441e-03 * T + 1.38464619e-05 * T2 -
                 9.62579362e-09 * T3 + 2.51970581e-12 * T4;
    // species 1: N
    species[1] = +1.50000000e+00;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    species[0] = +5.87712406e+05 * invT2 - 2.23924907e+03 * invT +
                 5.06694922e+00 - 6.13968550e-04 * T + 1.49180668e-07 * T2 -
                 1.92310549e-11 * T3 + 1.06195439e-15 * T4;
    // species 1: N
    species[1] = +8.87650138e+04 * invT2 - 1.07123150e+02 * invT +
                 1.36218829e+00 + 2.91672008e-04 * T - 1.72951510e-07 * T2 +
                 4.01265788e-11 * T3 - 2.67722757e-15 * T4;
  } else {
    // species 0: N2
    species[0] = +8.31013916e+08 * invT2 - 6.42073354e+05 * invT +
                 2.01026464e+02 - 3.06509205e-02 * T + 2.48690333e-06 * T2 -
                 9.70595411e-11 * T3 + 1.43753888e-15 * T4;
    // species 1: N
    species[1] = +5.47518105e+08 * invT2 - 3.10757498e+05 * invT +
                 6.81678274e+01 - 6.84798813e-03 * T + 3.82757240e-07 * T2 -
                 1.09836771e-11 * T3 + 1.27798602e-16 * T4;
  }
}

// compute Cp/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cp_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = +2.21037150e+04 * invT2 - 3.81846182e+02 * invT +
                 6.08273836e+00 - 8.53091441e-03 * T + 1.38464619e-05 * T2 -
                 9.62579362e-09 * T3 + 2.51970581e-12 * T4;
    // species 1: N
    species[1] = +2.50000000e+00;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    species[0] = +5.87712406e+05 * invT2 - 2.23924907e+03 * invT +
                 6.06694922e+00 - 6.13968550e-04 * T + 1.49180668e-07 * T2 -
                 1.92310549e-11 * T3 + 1.06195439e-15 * T4;
    // species 1: N
    species[1] = +8.87650138e+04 * invT2 - 1.07123150e+02 * invT +
                 2.36218829e+00 + 2.91672008e-04 * T - 1.72951510e-07 * T2 +
                 4.01265788e-11 * T3 - 2.67722757e-15 * T4;
  } else {
    // species 0: N2
    species[0] = +8.31013916e+08 * invT2 - 6.42073354e+05 * invT +
                 2.02026464e+02 - 3.06509205e-02 * T + 2.48690333e-06 * T2 -
                 9.70595411e-11 * T3 + 1.43753888e-15 * T4;
    // species 1: N
    species[1] = +5.47518105e+08 * invT2 - 3.10757498e+05 * invT +
                 6.91678274e+01 - 6.84798813e-03 * T + 3.82757240e-07 * T2 -
                 1.09836771e-11 * T3 + 1.27798602e-16 * T4;
  }
}

// compute the g/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
gibbs(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = -1.105185748500000e+04 * invT2 + 3.289999040000000e+02 * invT -
                 3.818461820000000e+02 * logT * invT -
                 6.082738360000000e+00 * logT + 1.684277580000000e+01 +
                 4.265457205000000e-03 * T - 2.307743648333333e-06 * T2 +
                 8.021494683333333e-10 * T3 - 1.259852904500000e-13 * T4;
    // species 1: N
    species[1] = +5.610463780000000e+04 * invT - 2.500000000000000e+00 * logT -
                 1.693905036000000e+00;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    species[0] = -2.938562030000000e+05 * invT2 + 1.059285507700000e+04 * invT -
                 2.239249073000000e+03 * logT * invT -
                 6.066949220000000e+00 * logT + 2.193334949000000e+01 +
                 3.069842750000000e-04 * T - 2.486344465000000e-08 * T2 +
                 1.602587904166667e-12 * T3 - 5.309771930000000e-17 * T4;
    // species 1: N
    species[1] = -4.438250690000000e+04 * invT2 + 5.686639015000000e+04 * invT -
                 1.071231500000000e+02 * logT * invT -
                 2.362188287000000e+00 * logT - 2.503043219000000e+00 -
                 1.458360040500000e-04 * T + 2.882525166666667e-08 * T2 -
                 3.343881566666666e-12 * T3 + 1.338613785500000e-16 * T4;
  } else {
    // species 0: N2
    species[0] = -4.155069580000000e+08 * invT2 + 4.296633686000000e+06 * invT -
                 6.420733540000001e+05 * logT * invT -
                 2.020264635000000e+02 * logT + 1.874126203500000e+03 +
                 1.532546023000000e-02 * T - 4.144838888333333e-07 * T2 +
                 8.088295091666667e-12 * T3 - 7.187694405000000e-17 * T4;
    // species 1: N
    species[1] = -2.737590525000000e+08 * invT2 + 2.239828120000000e+06 * invT -
                 3.107574980000000e+05 * logT * invT -
                 6.916782739999999e+01 * logT + 6.540448027000000e+02 +
                 3.423994065000000e-03 * T - 6.379287333333334e-08 * T2 +
                 9.153064241666666e-13 * T3 - 6.389930120000000e-18 * T4;
  }
}

// compute the a/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
helmholtz(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = -1.10518575e+04 * invT2 + 3.28999904e+02 * invT -
                 3.81846182e+02 * logT * invT - 6.08273836e+00 * logT +
                 1.58427758e+01 + 4.26545720e-03 * T - 2.30774365e-06 * T2 +
                 8.02149468e-10 * T3 - 1.25985290e-13 * T4;
    // species 1: N
    species[1] =
      +5.61046378e+04 * invT - 2.50000000e+00 * logT - 2.69390504e+00;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    species[0] = -2.93856203e+05 * invT2 + 1.05928551e+04 * invT -
                 2.23924907e+03 * logT * invT - 6.06694922e+00 * logT +
                 2.09333495e+01 + 3.06984275e-04 * T - 2.48634447e-08 * T2 +
                 1.60258790e-12 * T3 - 5.30977193e-17 * T4;
    // species 1: N
    species[1] = -4.43825069e+04 * invT2 + 5.68663901e+04 * invT -
                 1.07123150e+02 * logT * invT - 2.36218829e+00 * logT -
                 3.50304322e+00 - 1.45836004e-04 * T + 2.88252517e-08 * T2 -
                 3.34388157e-12 * T3 + 1.33861379e-16 * T4;
  } else {
    // species 0: N2
    species[0] = -4.15506958e+08 * invT2 + 4.29663369e+06 * invT -
                 6.42073354e+05 * logT * invT - 2.02026464e+02 * logT +
                 1.87312620e+03 + 1.53254602e-02 * T - 4.14483889e-07 * T2 +
                 8.08829509e-12 * T3 - 7.18769440e-17 * T4;
    // species 1: N
    species[1] = -2.73759052e+08 * invT2 + 2.23982812e+06 * invT -
                 3.10757498e+05 * logT * invT - 6.91678274e+01 * logT +
                 6.53044803e+02 + 3.42399406e-03 * T - 6.37928733e-08 * T2 +
                 9.15306424e-13 * T3 - 6.38993012e-18 * T4;
  }
}

// compute the e/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesInternalEnergy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = -2.21037150e+04 * invT2 - 3.81846182e+02 * logT * invT +
                 5.08273836e+00 - 4.26545720e-03 * T + 4.61548730e-06 * T2 -
                 2.40644840e-09 * T3 + 5.03941162e-13 * T4 +
                 7.10846086e+02 * invT;
    // species 1: N
    species[1] = +1.50000000e+00 + 5.61046378e+04 * invT;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    species[0] = -5.87712406e+05 * invT2 - 2.23924907e+03 * logT * invT +
                 5.06694922e+00 - 3.06984275e-04 * T + 4.97268893e-08 * T2 -
                 4.80776371e-12 * T3 + 2.12390877e-16 * T4 +
                 1.28321041e+04 * invT;
    // species 1: N
    species[1] = -8.87650138e+04 * invT2 - 1.07123150e+02 * logT * invT +
                 1.36218829e+00 + 1.45836004e-04 * T - 5.76505033e-08 * T2 +
                 1.00316447e-11 * T3 - 5.35445514e-16 * T4 +
                 5.69735133e+04 * invT;
  } else {
    // species 0: N2
    species[0] = -8.31013916e+08 * invT2 - 6.42073354e+05 * logT * invT +
                 2.01026464e+02 - 1.53254602e-02 * T + 8.28967778e-07 * T2 -
                 2.42648853e-11 * T3 + 2.87507776e-16 * T4 +
                 4.93870704e+06 * invT;
    // species 1: N
    species[1] = -5.47518105e+08 * invT2 - 3.10757498e+05 * logT * invT +
                 6.81678274e+01 - 3.42399406e-03 * T + 1.27585747e-07 * T2 -
                 2.74591927e-12 * T3 + 2.55597205e-17 * T4 +
                 2.55058562e+06 * invT;
  }
}

// compute the h/(RT) at the given temperature (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEnthalpy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = -2.21037150e+04 * invT2 - 3.81846182e+02 * logT * invT +
                 6.08273836e+00 - 4.26545720e-03 * T + 4.61548730e-06 * T2 -
                 2.40644840e-09 * T3 + 5.03941162e-13 * T4 +
                 7.10846086e+02 * invT;
    // species 1: N
    species[1] = +2.50000000e+00 + 5.61046378e+04 * invT;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    species[0] = -5.87712406e+05 * invT2 - 2.23924907e+03 * logT * invT +
                 6.06694922e+00 - 3.06984275e-04 * T + 4.97268893e-08 * T2 -
                 4.80776371e-12 * T3 + 2.12390877e-16 * T4 +
                 1.28321041e+04 * invT;
    // species 1: N
    species[1] = -8.87650138e+04 * invT2 - 1.07123150e+02 * logT * invT +
                 2.36218829e+00 + 1.45836004e-04 * T - 5.76505033e-08 * T2 +
                 1.00316447e-11 * T3 - 5.35445514e-16 * T4 +
                 5.69735133e+04 * invT;
  } else {
    // species 0: N2
    species[0] = -8.31013916e+08 * invT2 - 6.42073354e+05 * logT * invT +
                 2.02026464e+02 - 1.53254602e-02 * T + 8.28967778e-07 * T2 -
                 2.42648853e-11 * T3 + 2.87507776e-16 * T4 +
                 4.93870704e+06 * invT;
    // species 1: N
    species[1] = -5.47518105e+08 * invT2 - 3.10757498e+05 * logT * invT +
                 6.91678274e+01 - 3.42399406e-03 * T + 1.27585747e-07 * T2 -
                 2.74591927e-12 * T3 + 2.55597205e-17 * T4 +
                 2.55058562e+06 * invT;
  }
}

// compute the S/R at the given temperature (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEntropy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = -1.10518575e+04 * invT2 + 3.81846182e+02 * invT +
                 6.08273836e+00 * logT - 8.53091441e-03 * T +
                 6.92323094e-06 * T2 - 3.20859787e-09 * T3 +
                 6.29926452e-13 * T4 - 1.07600374e+01;
    // species 1: N
    species[1] = +2.50000000e+00 * logT + 4.19390504e+00;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    species[0] = -2.93856203e+05 * invT2 + 2.23924907e+03 * invT +
                 6.06694922e+00 * logT - 6.13968550e-04 * T +
                 7.45903340e-08 * T2 - 6.41035162e-12 * T3 +
                 2.65488596e-16 * T4 - 1.58664003e+01;
    // species 1: N
    species[1] = -4.43825069e+04 * invT2 + 1.07123150e+02 * invT +
                 2.36218829e+00 * logT + 2.91672008e-04 * T -
                 8.64757550e-08 * T2 + 1.33755263e-11 * T3 -
                 6.69306893e-16 * T4 + 4.86523151e+00;
  } else {
    // species 0: N2
    species[0] = -4.15506958e+08 * invT2 + 6.42073354e+05 * invT +
                 2.02026464e+02 * logT - 3.06509205e-02 * T +
                 1.24345167e-06 * T2 - 3.23531804e-11 * T3 +
                 3.59384720e-16 * T4 - 1.67209974e+03;
    // species 1: N
    species[1] = -2.73759052e+08 * invT2 + 3.10757498e+05 * invT +
                 6.91678274e+01 * logT - 6.84798813e-03 * T +
                 1.91378620e-07 * T2 - 3.66122570e-12 * T3 +
                 3.19496506e-17 * T4 - 5.84876975e+02;
  }
}

// compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
dcvpRdT(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real invT3 = invT * invT * invT;

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = -4.42074299e+04 * invT3 + 3.81846182e+02 * invT2 -
                 8.53091441e-03 + 2.76929238e-05 * T - 2.88773809e-08 * T2 +
                 1.00788232e-11 * T3;
    // species 1: N
    species[1] = 0.0;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    species[0] = -1.17542481e+06 * invT3 + 2.23924907e+03 * invT2 -
                 6.13968550e-04 + 2.98361336e-07 * T - 5.76931646e-11 * T2 +
                 4.24781754e-15 * T3;
    // species 1: N
    species[1] = -1.77530028e+05 * invT3 + 1.07123150e+02 * invT2 +
                 2.91672008e-04 - 3.45903020e-07 * T + 1.20379736e-10 * T2 -
                 1.07089103e-14 * T3;
  } else {
    // species 0: N2
    species[0] = -1.66202783e+09 * invT3 + 6.42073354e+05 * invT2 -
                 3.06509205e-02 + 4.97380667e-06 * T - 2.91178623e-10 * T2 +
                 5.75015552e-15 * T3;
    // species 1: N
    species[1] = -1.09503621e+09 * invT3 + 3.10757498e+05 * invT2 -
                 6.84798813e-03 + 7.65514480e-07 * T - 3.29510313e-11 * T2 +
                 5.11194410e-16 * T3;
  }
}

// Returns the mean specific heat at CP (Eq. 33)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cpbl)
{
  amrex::Real result = 0;
  amrex::Real cpor[2]; //  temporary storage
  cp_R(cpor, T);

  // perform dot product
  for (int id = 0; id < 2; ++id) {
    result += x[id] * cpor[id];
  }

  cpbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CP (Eq. 34)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cpbs)
{
  amrex::Real result = 0.0;

  // compute Cp/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    result += y[0] *
              (+2.21037150e+04 * invT2 - 3.81846182e+02 * invT +
               6.08273836e+00 - 8.53091441e-03 * T + 1.38464619e-05 * T2 -
               9.62579362e-09 * T3 + 2.51970581e-12 * T4) *
              0.0356964374955379;
    // species 1: N
    result += y[1] * (+2.50000000e+00) * 0.0713928749910759;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    result += y[0] *
              (+5.87712406e+05 * invT2 - 2.23924907e+03 * invT +
               6.06694922e+00 - 6.13968550e-04 * T + 1.49180668e-07 * T2 -
               1.92310549e-11 * T3 + 1.06195439e-15 * T4) *
              0.0356964374955379;
    // species 1: N
    result += y[1] *
              (+8.87650138e+04 * invT2 - 1.07123150e+02 * invT +
               2.36218829e+00 + 2.91672008e-04 * T - 1.72951510e-07 * T2 +
               4.01265788e-11 * T3 - 2.67722757e-15 * T4) *
              0.0713928749910759;
  } else {
    // species 0: N2
    result += y[0] *
              (+8.31013916e+08 * invT2 - 6.42073354e+05 * invT +
               2.02026464e+02 - 3.06509205e-02 * T + 2.48690333e-06 * T2 -
               9.70595411e-11 * T3 + 1.43753888e-15 * T4) *
              0.0356964374955379;
    // species 1: N
    result += y[1] *
              (+5.47518105e+08 * invT2 - 3.10757498e+05 * invT +
               6.91678274e+01 - 6.84798813e-03 * T + 3.82757240e-07 * T2 -
               1.09836771e-11 * T3 + 1.27798602e-16 * T4) *
              0.0713928749910759;
  }

  cpbs = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 35)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cvbl)
{
  amrex::Real result = 0;
  amrex::Real cvor[2]; //  temporary storage
  cv_R(cvor, T);

  // perform dot product
  for (int id = 0; id < 2; ++id) {
    result += x[id] * cvor[id];
  }

  cvbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 36)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cvbs)
{
  amrex::Real result = 0.0;
  // compute Cv/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    result += y[0] *
              (+2.21037150e+04 * invT2 - 3.81846182e+02 * invT +
               5.08273836e+00 - 8.53091441e-03 * T + 1.38464619e-05 * T2 -
               9.62579362e-09 * T3 + 2.51970581e-12 * T4) *
              0.0356964374955379;
    // species 1: N
    result += y[1] * (+1.50000000e+00) * 0.0713928749910759;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    result += y[0] *
              (+5.87712406e+05 * invT2 - 2.23924907e+03 * invT +
               5.06694922e+00 - 6.13968550e-04 * T + 1.49180668e-07 * T2 -
               1.92310549e-11 * T3 + 1.06195439e-15 * T4) *
              0.0356964374955379;
    // species 1: N
    result += y[1] *
              (+8.87650138e+04 * invT2 - 1.07123150e+02 * invT +
               1.36218829e+00 + 2.91672008e-04 * T - 1.72951510e-07 * T2 +
               4.01265788e-11 * T3 - 2.67722757e-15 * T4) *
              0.0713928749910759;
  } else {
    // species 0: N2
    result += y[0] *
              (+8.31013916e+08 * invT2 - 6.42073354e+05 * invT +
               2.01026464e+02 - 3.06509205e-02 * T + 2.48690333e-06 * T2 -
               9.70595411e-11 * T3 + 1.43753888e-15 * T4) *
              0.0356964374955379;
    // species 1: N
    result += y[1] *
              (+5.47518105e+08 * invT2 - 3.10757498e+05 * invT +
               6.81678274e+01 - 6.84798813e-03 * T + 3.82757240e-07 * T2 -
               1.09836771e-11 * T3 + 1.27798602e-16 * T4) *
              0.0713928749910759;
  }

  cvbs = result * 8.31446261815324e+07;
}

// Returns the mean enthalpy of the mixture in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBML(const amrex::Real T, const amrex::Real x[], amrex::Real& hbml)
{
  amrex::Real result = 0;
  amrex::Real hml[2];                        //  temporary storage
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // perform dot product
  for (int id = 0; id < 2; ++id) {
    result += x[id] * hml[id];
  }

  hbml = result * RT;
}

// Returns mean enthalpy of mixture in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& hbms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    result +=
      y[0] *
      (-2.21037150e+04 * invT2 - 3.81846182e+02 * logT * invT + 6.08273836e+00 -
       4.26545720e-03 * T + 4.61548730e-06 * T2 - 2.40644840e-09 * T3 +
       5.03941162e-13 * T4 + 7.10846086e+02 * invT) *
      0.0356964374955379;
    // species 1: N
    result +=
      y[1] * (+2.50000000e+00 + 5.61046378e+04 * invT) * 0.0713928749910759;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    result +=
      y[0] *
      (-5.87712406e+05 * invT2 - 2.23924907e+03 * logT * invT + 6.06694922e+00 -
       3.06984275e-04 * T + 4.97268893e-08 * T2 - 4.80776371e-12 * T3 +
       2.12390877e-16 * T4 + 1.28321041e+04 * invT) *
      0.0356964374955379;
    // species 1: N
    result +=
      y[1] *
      (-8.87650138e+04 * invT2 - 1.07123150e+02 * logT * invT + 2.36218829e+00 +
       1.45836004e-04 * T - 5.76505033e-08 * T2 + 1.00316447e-11 * T3 -
       5.35445514e-16 * T4 + 5.69735133e+04 * invT) *
      0.0713928749910759;
  } else {
    // species 0: N2
    result +=
      y[0] *
      (-8.31013916e+08 * invT2 - 6.42073354e+05 * logT * invT + 2.02026464e+02 -
       1.53254602e-02 * T + 8.28967778e-07 * T2 - 2.42648853e-11 * T3 +
       2.87507776e-16 * T4 + 4.93870704e+06 * invT) *
      0.0356964374955379;
    // species 1: N
    result +=
      y[1] *
      (-5.47518105e+08 * invT2 - 3.10757498e+05 * logT * invT + 6.91678274e+01 -
       3.42399406e-03 * T + 1.27585747e-07 * T2 - 2.74591927e-12 * T3 +
       2.55597205e-17 * T4 + 2.55058562e+06 * invT) *
      0.0713928749910759;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  hbms = result * RT;
}

// get mean internal energy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBML(const amrex::Real T, const amrex::Real x[], amrex::Real& ubml)
{
  amrex::Real result = 0;
  amrex::Real uml[2];                        //  temporary energy array
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // perform dot product
  for (int id = 0; id < 2; ++id) {
    result += x[id] * uml[id];
  }

  ubml = result * RT;
}

// get mean internal energy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& ubms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    result +=
      y[0] *
      (-2.21037150e+04 * invT2 - 3.81846182e+02 * logT * invT + 5.08273836e+00 -
       4.26545720e-03 * T + 4.61548730e-06 * T2 - 2.40644840e-09 * T3 +
       5.03941162e-13 * T4 + 7.10846086e+02 * invT) *
      0.0356964374955379;
    // species 1: N
    result +=
      y[1] * (+1.50000000e+00 + 5.61046378e+04 * invT) * 0.0713928749910759;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    result +=
      y[0] *
      (-5.87712406e+05 * invT2 - 2.23924907e+03 * logT * invT + 5.06694922e+00 -
       3.06984275e-04 * T + 4.97268893e-08 * T2 - 4.80776371e-12 * T3 +
       2.12390877e-16 * T4 + 1.28321041e+04 * invT) *
      0.0356964374955379;
    // species 1: N
    result +=
      y[1] *
      (-8.87650138e+04 * invT2 - 1.07123150e+02 * logT * invT + 1.36218829e+00 +
       1.45836004e-04 * T - 5.76505033e-08 * T2 + 1.00316447e-11 * T3 -
       5.35445514e-16 * T4 + 5.69735133e+04 * invT) *
      0.0713928749910759;
  } else {
    // species 0: N2
    result +=
      y[0] *
      (-8.31013916e+08 * invT2 - 6.42073354e+05 * logT * invT + 2.01026464e+02 -
       1.53254602e-02 * T + 8.28967778e-07 * T2 - 2.42648853e-11 * T3 +
       2.87507776e-16 * T4 + 4.93870704e+06 * invT) *
      0.0356964374955379;
    // species 1: N
    result +=
      y[1] *
      (-5.47518105e+08 * invT2 - 3.10757498e+05 * logT * invT + 6.81678274e+01 -
       3.42399406e-03 * T + 1.27585747e-07 * T2 - 2.74591927e-12 * T3 +
       2.55597205e-17 * T4 + 2.55058562e+06 * invT) *
      0.0713928749910759;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  ubms = result * RT;
}

// get mixture entropy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBML(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& sbml)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[2]; //  temporary storage
  speciesEntropy(sor, T);

  // Compute Eq 42
  for (int id = 0; id < 2; ++id) {
    result += x[id] * (sor[id] - log((x[id] + 1e-100)) - logPratio);
  }

  sbml = result * 8.31446261815324e+07;
}

// get mixture entropy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBMS(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& sbms)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[2];  //  temporary storage
  amrex::Real x[2];    //  need a ytx conversion
  amrex::Real YOW = 0; // See Eq 4, 6 in CK Manual

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 2; i++) {
    YOW += y[i] * imw(i);
  }
  // Now compute y to x conversion
  x[0] = y[0] / (28.014000 * YOW);
  x[1] = y[1] / (14.007000 * YOW);
  speciesEntropy(sor, T);
  // Perform computation in Eq 42 and 43
  for (int i = 0; i < 2; i++) {
    result += x[i] * (sor[i] - log((x[i] + 1e-100)) - logPratio);
  }
  // Scale by R/W
  sbms = result * 8.31446261815324e+07 * YOW;
}

//  get temperature given internal energy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_EY(
  const amrex::Real e, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 90;   // max lower bound for thermo def
  amrex::Real tmax = 4000; // min upper bound for thermo def
  amrex::Real e1, emin, emax, cv, t1, dt;
  CKUBMS(tmin, y, emin);
  CKUBMS(tmax, y, emax);
  if (e < emin) {
    // Linear Extrapolation below tmin
    CKCVBS(tmin, y, cv);
    t = tmin - (emin - e) / cv;
    ierr = 1;
    return;
  }
  if (e > emax) {
    // Linear Extrapolation above tmax
    CKCVBS(tmax, y, cv);
    t = tmax - (emax - e) / cv;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (emax - emin) * (e - emin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKUBMS(t1, y, e1);
    CKCVBS(t1, y, cv);
    dt = (e - e1) / cv;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

//  get temperature given enthalpy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_HY(
  const amrex::Real h, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 90;   // max lower bound for thermo def
  amrex::Real tmax = 4000; // min upper bound for thermo def
  amrex::Real h1, hmin, hmax, cp, t1, dt;
  CKHBMS(tmin, y, hmin);
  CKHBMS(tmax, y, hmax);
  if (h < hmin) {
    // Linear Extrapolation below tmin
    CKCPBS(tmin, y, cp);
    t = tmin - (hmin - h) / cp;
    ierr = 1;
    return;
  }
  if (h > hmax) {
    // Linear Extrapolation above tmax
    CKCPBS(tmax, y, cp);
    t = tmax - (hmax - h) / cp;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (hmax - hmin) * (h - hmin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKHBMS(t1, y, h1);
    CKCPBS(t1, y, cp);
    dt = (h - h1) / cp;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

// Compute P = rhoRT/W(x)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPX(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& P)
{
  amrex::Real XW = 0;                      //  To hold mean molecular wt
  XW += x[0] * 28.014000;                  // N2
  XW += x[1] * 14.007000;                  // N
  P = rho * 8.31446261815324e+07 * T / XW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(y)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPY(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& P)
{
  amrex::Real YOW = 0; //  for computing mean MW

  for (int i = 0; i < 2; i++) {
    YOW += y[i] * imw(i);
  }
  P = rho * 8.31446261815324e+07 * T * YOW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(c)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPC(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& P)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 28.014000; // N2
  W += c[1] * 14.007000; // N

  for (int id = 0; id < 2; ++id) {
    sumC += c[id];
  }
  P = rho * 8.31446261815324e+07 * T * sumC / W; // P = rho*R*T/W
}

// Compute rho = PW(x)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOX(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& rho)
{
  amrex::Real XW = 0;                        //  To hold mean molecular wt
  XW += x[0] * 28.014000;                    // N2
  XW += x[1] * 14.007000;                    // N
  rho = P * XW / (8.31446261815324e+07 * T); // rho = P*W/(R*T)
}

// Compute rho = P*W(y)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOY(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& rho)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 2; i++) {
    YOW += y[i] * imw(i);
  }

  rho = P / (8.31446261815324e+07 * T * YOW); // rho = P*W/(R*T)
}

// Compute rho = P*W(c)/(R*T)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOC(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& rho)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 28.014000; // N2
  W += c[1] * 14.007000; // N

  for (int id = 0; id < 2; ++id) {
    sumC += c[id];
  }
  rho = P * W / (sumC * T * 8.31446261815324e+07); // rho = PW/(R*T)
}

// get molecular weight for all species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWT(amrex::Real wt[])
{
  get_mw(wt);
}

// given y[species]: mass fractions
// s mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWY(const amrex::Real y[], amrex::Real& wtm)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 2; i++) {
    YOW += y[i] * imw(i);
  }

  wtm = 1.0 / YOW;
}

// given x[species]: mole fractions
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWX(const amrex::Real x[], amrex::Real& wtm)
{
  amrex::Real XW = 0;     //  see Eq 4 in CK Manual
  XW += x[0] * 28.014000; // N2
  XW += x[1] * 14.007000; // N
  wtm = XW;
}

// given c[species]: molar concentration
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWC(const amrex::Real c[], amrex::Real& wtm)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 28.014000; // N2
  W += c[1] * 14.007000; // N

  for (int id = 0; id < 2; ++id) {
    sumC += c[id];
  }
  //  CK provides no guard against division by zero
  wtm = W / sumC;
}

// get Cp/R as a function of T
// for all species (Eq 19)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPOR(const amrex::Real T, amrex::Real cpor[])
{
  cp_R(cpor, T);
}

// get H/RT as a function of T
// for all species (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHORT(const amrex::Real T, amrex::Real hort[])
{
  speciesEnthalpy(hort, T);
}

// get S/R as a function of T
// for all species (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSOR(const amrex::Real T, amrex::Real sor[])
{
  speciesEntropy(sor, T);
}

// convert y[species] (mass fracs) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTX(const amrex::Real y[], amrex::Real x[])
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 2; i++) {
    YOW += y[i] * imw(i);
  }

  amrex::Real YOWINV = 1.0 / YOW;

  for (int i = 0; i < 2; i++) {
    x[i] = y[i] * imw(i) * YOWINV;
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real c[])
{
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 2; i++) {
    c[i] = y[i] * imw(i);
  }
  for (int i = 0; i < 2; i++) {
    YOW += c[i];
  }

  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // Now compute conversion

  for (int i = 0; i < 2; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCR(
  const amrex::Real rho,
  amrex::Real /*T*/,
  const amrex::Real y[],
  amrex::Real c[])
{

  for (int i = 0; i < 2; i++) {
    c[i] = rho * y[i] * imw(i);
  }
}

// convert x[species] (mole fracs) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTY(const amrex::Real x[], amrex::Real y[])
{
  amrex::Real XW = 0; // See Eq 4, 9 in CK Manual
  // Compute mean molecular wt first
  XW += x[0] * 28.014000; // N2
  XW += x[1] * 14.007000; // N
  // Now compute conversion
  amrex::Real XWinv = 1.0 / XW;
  y[0] = x[0] * 28.014000 * XWinv;
  y[1] = x[1] * 14.007000 * XWinv;
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real PORT = P / (8.31446261815324e+07 * T); // P/RT

  // Compute conversion, see Eq 10
  for (int id = 0; id < 2; ++id) {
    c[id] = x[id] * PORT;
  }
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCR(
  const amrex::Real rho,
  const amrex::Real /*T*/,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 28.014000; // N2
  XW += x[1] * 14.007000; // N
  ROW = rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 2; ++id) {
    c[id] = x[id] * ROW;
  }
}

// convert c[species] (molar conc) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTX(const amrex::Real c[], amrex::Real x[])
{
  amrex::Real sumC = 0;

  // compute sum of c
  for (int id = 0; id < 2; ++id) {
    sumC += c[id];
  }

  //  See Eq 13
  amrex::Real sumCinv = 1.0 / sumC;
  for (int id = 0; id < 2; ++id) {
    x[id] = c[id] * sumCinv;
  }
}

// convert c[species] (molar conc) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTY(const amrex::Real c[], amrex::Real y[])
{
  amrex::Real CW = 0; // See Eq 12 in CK Manual
  // compute denominator in eq 12 first
  CW += c[0] * 28.014000; // N2
  CW += c[1] * 14.007000; // N
  // Now compute conversion
  amrex::Real CWinv = 1.0 / CW;
  y[0] = c[0] * 28.014000 * CWinv;
  y[1] = c[1] * 14.007000 * CWinv;
}

// get specific heat at constant volume as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVML(const amrex::Real T, amrex::Real cvml[])
{
  cv_R(cvml, T);

  // convert to chemkin units
  for (int id = 0; id < 2; ++id) {
    cvml[id] *= 8.31446261815324e+07;
  }
}

// get specific heat at constant pressure as a
// function of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPML(const amrex::Real T, amrex::Real cpml[])
{
  cp_R(cpml, T);

  // convert to chemkin units
  for (int id = 0; id < 2; ++id) {
    cpml[id] *= 8.31446261815324e+07;
  }
}

// get internal energy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUML(const amrex::Real T, amrex::Real uml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // convert to chemkin units
  for (int id = 0; id < 2; ++id) {
    uml[id] *= RT;
  }
}

// get enthalpy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHML(const amrex::Real T, amrex::Real hml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // convert to chemkin units
  for (int id = 0; id < 2; ++id) {
    hml[id] *= RT;
  }
}

// Returns the standard-state entropies in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSML(const amrex::Real T, amrex::Real sml[])
{
  speciesEntropy(sml, T);

  // convert to chemkin units
  for (int id = 0; id < 2; ++id) {
    sml[id] *= 8.31446261815324e+07;
  }
}

// Returns the specific heats at constant volume
// in mass units (Eq. 29)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVMS(const amrex::Real T, amrex::Real cvms[])
{
  cv_R(cvms, T);
  // multiply by R/molecularweight
  cvms[0] *= 2.967966951578939e+06; // N2
  cvms[1] *= 5.935933903157878e+06; // N
}

// Returns the specific heats at constant pressure
// in mass units (Eq. 26)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPMS(const amrex::Real T, amrex::Real cpms[])
{
  cp_R(cpms, T);
  // multiply by R/molecularweight
  cpms[0] *= 2.967966951578939e+06; // N2
  cpms[1] *= 5.935933903157878e+06; // N
}

// Returns internal energy in mass units (Eq 30.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUMS(const amrex::Real T, amrex::Real ums[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesInternalEnergy(ums, T);

  for (int i = 0; i < 2; i++) {
    ums[i] *= RT * imw(i);
  }
}

// Returns enthalpy in mass units (Eq 27.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHMS(const amrex::Real T, amrex::Real hms[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesEnthalpy(hms, T);

  for (int i = 0; i < 2; i++) {
    hms[i] *= RT * imw(i);
  }
}

// Returns the entropies in mass units (Eq 28.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSMS(const amrex::Real T, amrex::Real sms[])
{
  speciesEntropy(sms, T);
  // multiply by R/molecularweight
  sms[0] *= 2.967966951578939e+06; // N2
  sms[1] *= 5.935933903157878e+06; // N
}

// GPU version of productionRate: no more use of thermo namespace vectors
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
comp_qfqr(
  amrex::Real* qf,
  amrex::Real* qr,
  const amrex::Real* sc,
  const amrex::Real* /*sc_qss*/,
  const amrex::Real T,
  const amrex::Real invT,
  const amrex::Real logT)
{

  // reaction 0: N2 + M <=> 2 N + M
  qf[0] = sc[0];
  qr[0] = (sc[1] * sc[1]);

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 2; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[2];
  gibbs(g_RT, T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 * invT;
  amrex::Real refCinv = 1 / refC;

  // Evaluate the kfs
  amrex::Real k_f, Corr;

  // reaction 0:  N2 + M <=> 2 N + M
  k_f = 112000000 * exp(-(97925.9631655985) * invT);
  Corr = mixture;
  qf[0] *= Corr * k_f;
  qr[0] *= Corr * k_f * exp(-(g_RT[0] - 2.000000 * g_RT[1])) * (refCinv);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
productionRate(amrex::Real* wdot, const amrex::Real* sc, const amrex::Real T)
{
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  const amrex::Real refC = 101325 / 8.31446 * invT;
  const amrex::Real refCinv = 1 / refC;

  for (int i = 0; i < 2; ++i) {
    wdot[i] = 0.0;
  }

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 2; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[2];
  gibbs(g_RT, T);

  {
    // reaction 0:  N2 + M <=> 2 N + M
    const amrex::Real k_f = 112000000 * exp(-(97925.9631655985) * invT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[0]);
    const amrex::Real qr = Corr * k_f * exp(-(g_RT[0] - 2.000000 * g_RT[1])) *
                           (refCinv) * ((sc[1] * sc[1]));
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[1] += 2.000000 * qdot;
  }
}

// compute the production rate for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWC(const amrex::Real T, amrex::Real C[], amrex::Real wdot[])
{

  // convert to SI
  for (int id = 0; id < 2; ++id) {
    C[id] *= 1.0e6;
  }

  // convert to chemkin units
  productionRate(wdot, C, T);

  // convert to chemkin units
  for (int id = 0; id < 2; ++id) {
    C[id] *= 1.0e-6;
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given P, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWYP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real wdot[])
{
  amrex::Real c[2]; // temporary storage
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 2; i++) {
    YOW += y[i] * imw(i);
  }
  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // multiply by 1e6 so c goes to SI
  PWORT *= 1e6;
  // Now compute conversion (and go to SI)
  for (int i = 0; i < 2; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 2; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given P, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWXP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real wdot[])
{
  amrex::Real c[2]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 2; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 2; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given rho, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWYR(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real wdot[])
{
  amrex::Real c[2]; // temporary storage

  // See Eq 8 with an extra 1e6 so c goes to SI
  for (int i = 0; i < 2; i++) {
    c[i] = 1e6 * rho * y[i] * imw(i);
  }

  // call productionRate
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 2; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given rho, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWXR(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real wdot[])
{
  amrex::Real c[2];   // temporary storage
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 28.014000; // N2
  XW += x[1] * 14.007000; // N
  // Extra 1e6 factor to take c to SI
  ROW = 1e6 * rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 2; ++id) {
    c[id] = x[id] * ROW;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 2; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRG(int kcharge[])
{
  kcharge[0] = 0; // N2
  kcharge[1] = 0; // N
}

//  species charge per unit mass
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRGMASS(amrex::Real zk[])
{

  int kchrg[2];
  CKCHRG(kchrg);

  for (int id = 0; id < 2; ++id) {
    zk[id] = 6.02214076e+23 * 1.60217663e-19 * kchrg[id] * imw(id);
  }
}

// compute an approx to the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian_precond(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int HP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 9; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[2];
  for (auto& val : wdot) {
    val = 0.0;
  }

  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 / T;
  amrex::Real refCinv = 1.0 / refC;

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int k = 0; k < 2; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[2];
  gibbs(g_RT, T);

  // compute the species enthalpy
  amrex::Real h_RT[2];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[2];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 0: N2 + M <=> 2 N + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[0];
  k_f = 112000000 * exp(-(97925.9631655985) * invT);
  dlnkfdT = (97925.9631655985) * invT2;
  // reverse
  phi_r = (sc[1] * sc[1]);
  Kc = refC * exp(g_RT[0] - 2.000000 * g_RT[1]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0]) + (2.000000 * h_RT[1]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q;     // N2
  wdot[1] += 2 * q; // N
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor + k_f;
  dqdc[1] = q_nocor - k_r * 2.000000 * sc[1];
  for (int k = 0; k < 2; k++) {
    J[3 * k + 0] -= dqdc[k];
    J[3 * k + 1] += 2 * dqdc[k];
  }
  J[6] -= dqdT;     // dwdot[N2]/dT
  J[7] += 2 * dqdT; // dwdot[N]/dT

  amrex::Real c_R[2], dcRdT[2], e_RT[2];
  amrex::Real* eh_RT;
  if (HP == 1) {
    cp_R(c_R, T);
    dcvpRdT(dcRdT, T);
    eh_RT = &h_RT[0];
  } else {
    cv_R(c_R, T);
    dcvpRdT(dcRdT, T);
    speciesInternalEnergy(e_RT, T);
    eh_RT = &e_RT[0];
  }

  amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT = 0.0, dehmixdT = 0.0;
  for (int k = 0; k < 2; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[6 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 2; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 2; ++m) {
      dehmixdc += eh_RT[m] * J[k * 3 + m];
    }
    J[k * 3 + 2] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[8] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
#else
  amrex::Abort();
#endif
}

// compute an approx to the reaction Jacobian (for preconditioning)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT_SIMPLIFIED(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real* Tp, const int* HP)
{
  amrex::Real c[2];

  for (int k = 0; k < 2; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian_precond(J, c, *Tp, *HP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 2; k++) {
    J[6 + k] *= 1.e-6;
    J[k * 3 + 2] *= 1.e6;
  }
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int consP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 9; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[2];
  for (auto& val : wdot) {
    val = 0.0;
  }

  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 / T;
  amrex::Real refCinv = 1.0 / refC;

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int k = 0; k < 2; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[2];
  gibbs(g_RT, T);

  // compute the species enthalpy
  amrex::Real h_RT[2];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[2];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 0: N2 + M <=> 2 N + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[0];
  k_f = 112000000 * exp(-(97925.9631655985) * invT);
  dlnkfdT = (97925.9631655985) * invT2;
  // reverse
  phi_r = (sc[1] * sc[1]);
  Kc = refC * exp(g_RT[0] - 2.000000 * g_RT[1]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0]) + (2.000000 * h_RT[1]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q;     // N2
  wdot[1] += 2 * q; // N
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[N2]
    dqdci = (1 - 1) * q_nocor + k_f;
    J[0] -= dqdci;     // dwdot[N2]/d[N2]
    J[1] += 2 * dqdci; // dwdot[N]/d[N2]
    // d()/d[N]
    dqdci = (1 - 1) * q_nocor - k_r * 2.000000 * sc[1];
    J[3] -= dqdci;     // dwdot[N2]/d[N]
    J[4] += 2 * dqdci; // dwdot[N]/d[N]
  } else {
    dqdc[0] = q_nocor + k_f;
    dqdc[1] = q_nocor - k_r * 2.000000 * sc[1];
    for (int k = 0; k < 2; k++) {
      J[3 * k + 0] -= dqdc[k];
      J[3 * k + 1] += 2 * dqdc[k];
    }
  }
  J[6] -= dqdT;     // dwdot[N2]/dT
  J[7] += 2 * dqdT; // dwdot[N]/dT

  amrex::Real c_R[2], dcRdT[2], e_RT[2];
  amrex::Real* eh_RT;
  if (consP == 1) {
    cp_R(c_R, T);
    dcvpRdT(dcRdT, T);
    eh_RT = &h_RT[0];
  } else {
    cv_R(c_R, T);
    dcvpRdT(dcRdT, T);
    speciesInternalEnergy(e_RT, T);
    eh_RT = &e_RT[0];
  }

  amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT = 0.0, dehmixdT = 0.0;
  for (int k = 0; k < 2; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[6 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 2; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 2; ++m) {
      dehmixdc += eh_RT[m] * J[k * 3 + m];
    }
    J[k * 3 + 2] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[8] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
#else
  amrex::Abort();
#endif
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT(
  amrex::Real* J,
  const amrex::Real* sc,
  const amrex::Real* Tp,
  const int* consP)
{
  amrex::Real c[2];

  for (int k = 0; k < 2; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian(J, c, *Tp, *consP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 2; k++) {
    J[6 + k] *= 1.e-6;
    J[k * 3 + 2] *= 1.e6;
  }
}

// Transport function declarations

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENIMC(int* LENIMC)
{
  *LENIMC = 8;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENRMC(int* LENRMC)
{
  *LENRMC = 130;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNO(int* NO)
{
  *NO = 4;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKK(int* KK)
{
  *KK = 2;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLITE(int* NLITE)
{
  *NLITE = 0;
}

// Patm in ergs/cm3
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPATM(amrex::Real* PATM)
{
  *PATM = 0.1013250000000000E+07;
}

// the molecular weights in g/mol
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetWT(amrex::Real* WT)
{
  WT[0] = 2.80140000E+01;
  WT[1] = 1.40070000E+01;
}

// the lennard-jones potential well depth eps/kb in K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetEPS(amrex::Real* EPS)
{
  EPS[0] = 9.75300000E+01;
  EPS[1] = 7.14000000E+01;
}

// the lennard-jones collision diameter in Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetSIG(amrex::Real* SIG)
{
  SIG[0] = 3.62100000E+00;
  SIG[1] = 3.29800000E+00;
}

// the dipole moment in Debye
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetDIP(amrex::Real* DIP)
{
  DIP[0] = 0.00000000E+00;
  DIP[1] = 0.00000000E+00;
}

// the polarizability in cubic Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPOL(amrex::Real* POL)
{
  POL[0] = 1.76000000E+00;
  POL[1] = 1.10000000E+00;
}

// the rotational relaxation collision number at 298 K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetZROT(amrex::Real* ZROT)
{
  ZROT[0] = 4.00000000E+00;
  ZROT[1] = 0.00000000E+00;
}

// 0: monoatomic, 1: linear, 2: nonlinear
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLIN(int* NLIN)
{
  NLIN[0] = 1;
  NLIN[1] = 0;
}

// Poly fits for the viscosities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFETA(amrex::Real* COFETA)
{
  COFETA[0] = -1.50463741E+01;
  COFETA[1] = 1.67877229E+00;
  COFETA[2] = -1.26116126E-01;
  COFETA[3] = 5.09682240E-03;
  COFETA[4] = -1.42328009E+01;
  COFETA[5] = 1.35693038E+00;
  COFETA[6] = -8.95108371E-02;
  COFETA[7] = 3.73378596E-03;
}

// Poly fits for the conductivities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFLAM(amrex::Real* COFLAM)
{
  COFLAM[0] = -7.33106218E+00;
  COFLAM[1] = 5.43841742E+00;
  COFLAM[2] = -6.59972954E-01;
  COFLAM[3] = 3.06382019E-02;
  COFLAM[4] = 2.68549072E+00;
  COFLAM[5] = 1.35693038E+00;
  COFLAM[6] = -8.95108371E-02;
  COFLAM[7] = 3.73378596E-03;
}

// Poly fits for the diffusion coefficients, dim NO*KK*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFD(amrex::Real* COFD)
{
  COFD[0] = -1.35708498E+01;
  COFD[1] = 2.60892593E+00;
  COFD[2] = -1.14790334E-01;
  COFD[3] = 4.57509943E-03;
  COFD[4] = -1.28326773E+01;
  COFD[5] = 2.46590522E+00;
  COFD[6] = -9.87792544E-02;
  COFD[7] = 3.98641293E-03;
  COFD[8] = -1.28326773E+01;
  COFD[9] = 2.46590522E+00;
  COFD[10] = -9.87792544E-02;
  COFD[11] = 3.98641293E-03;
  COFD[12] = -1.20872384E+01;
  COFD[13] = 2.29284058E+00;
  COFD[14] = -7.82697214E-02;
  COFD[15] = 3.18480527E-03;
}

// List of specs with small weight, dim NLITE
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKTDIF(int* /*KTDIF*/)
{
}

// Poly fits for thermal diff ratios, dim NO*NLITE*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFTD(amrex::Real* /*COFTD*/)
{
}

// compute the critical parameters for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS(
  amrex::Real* Tci, amrex::Real* ai, amrex::Real* bi, amrex::Real* acentric_i)
{

  amrex::Real EPS[2];
  amrex::Real SIG[2];
  amrex::Real wt[2];
  amrex::Real Rcst = 83.144598; // in bar [CGS] !
  amrex::Real avogadro = 6.02214199e23;
  amrex::Real boltzmann = 1.3806503e-16; // we work in CGS

  egtransetEPS(EPS);
  egtransetSIG(SIG);
  get_mw(wt);

  // species 0: N2
  // Imported from NIST
  Tci[0] = 126.192000;
  ai[0] = 1e6 * 0.42748 * Rcst * Rcst * Tci[0] * Tci[0] /
          (28.013400 * 28.013400 * 33.958000);
  bi[0] = 0.08664 * Rcst * Tci[0] / (28.013400 * 33.958000);
  acentric_i[0] = 0.037200;

  // species 1: N
  Tci[1] = 1.316 * EPS[1];
  ai[1] = (5.55 * avogadro * avogadro * EPS[1] * boltzmann * 1e-24 * SIG[1] *
           SIG[1] * SIG[1]) /
          (wt[1] * wt[1]);
  bi[1] = 0.855 * avogadro * 1e-24 * SIG[1] * SIG[1] * SIG[1] / (wt[1]);
  acentric_i[1] = 0.0;
}

// compute the critical parameter quantities for each species for SRK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS_SRK(
  amrex::Real* sqrtOneOverTc,
  amrex::Real* sqrtAsti,
  amrex::Real* Bi,
  amrex::Real* Fomega)
{

  // species 0: N2
  // Imported from NIST
  sqrtOneOverTc[0] = 8.9019282240563e-02;
  sqrtAsti[0] = 4.2022983929562e+04;
  Bi[0] = 9.5560052221563e-01;
  Fomega[0] = 5.4259343186608e-01;

  // species 1: N
  // Computed from Lennard-Jones
  sqrtOneOverTc[1] = 1.0316275924140e-01;
  sqrtAsti[1] = 6.0230890290396e+04;
  Bi[1] = 1.3186329724611e+00;
  Fomega[1] = 4.8508000000000e-01;
}

// gauss-jordan solver external routine
// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
sgjsolve(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/)
{
  amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
sgjsolve_simplified(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/)
{
  amrex::Abort(
    "sgjsolve_simplified not implemented, choose a different solver ");
}

#endif
