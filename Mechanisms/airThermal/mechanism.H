#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/* Elements
0  O
1  N
2  E
*/

// Species
#define N2_ID 0
#define N_ID 1
#define O_ID 2
#define O2_ID 3
#define NO_ID 4
#define N2O_ID 5
#define NO2_ID 6
#define N2O3_ID 7
#define NO3_ID 8

#define NUM_GAS_ELEMENTS 3   // Elements in the homogeneous phase
#define NUM_GAS_SPECIES 9    // Species in the homogeneous phase
#define NUM_GAS_REACTIONS 20 // Reactions in the homogeneous phase

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
  mm = 3;
  kk = 9;
  ii = 20;
  nfit = -1; // Why do you need this anyway ?
}

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[9] = {
  0.0356964374955379, // N2
  0.0713928749910759, // N
  0.0625039064941559, // O
  0.0312519532470779, // O2
  0.0333266679997334, // NO
  0.0227205598345943, // N2O
  0.0217367677426367, // NO2
  0.0131559905803107, // N2O3
  0.0161279917424682, // NO3
};
#endif
const amrex::Real h_global_imw[9] = {
  0.0356964374955379, // N2
  0.0713928749910759, // N
  0.0625039064941559, // O
  0.0312519532470779, // O2
  0.0333266679997334, // NO
  0.0227205598345943, // N2O
  0.0217367677426367, // NO2
  0.0131559905803107, // N2O3
  0.0161279917424682, // NO3
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[9] = {
  28.014000, // N2
  14.007000, // N
  15.999000, // O
  31.998000, // O2
  30.006000, // NO
  44.013000, // N2O
  46.005000, // NO2
  76.011000, // N2O3
  62.004000, // NO3
};
#endif
const amrex::Real h_global_mw[9] = {
  28.014000, // N2
  14.007000, // N
  15.999000, // O
  31.998000, // O2
  30.006000, // NO
  44.013000, // N2O
  46.005000, // NO2
  76.011000, // N2O3
  62.004000, // NO3
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_imw(amrex::Real* imw_new)
{
  imw_new[0] = 0.0356964374955379; // N2
  imw_new[1] = 0.0713928749910759; // N
  imw_new[2] = 0.0625039064941559; // O
  imw_new[3] = 0.0312519532470779; // O2
  imw_new[4] = 0.0333266679997334; // NO
  imw_new[5] = 0.0227205598345943; // N2O
  imw_new[6] = 0.0217367677426367; // NO2
  imw_new[7] = 0.0131559905803107; // N2O3
  imw_new[8] = 0.0161279917424682; // NO3
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
  mw_new[2] = 15.999000; // O
  mw_new[3] = 31.998000; // O2
  mw_new[4] = 30.006000; // NO
  mw_new[5] = 44.013000; // N2O
  mw_new[6] = 46.005000; // NO2
  mw_new[7] = 76.011000; // N2O3
  mw_new[8] = 62.004000; // NO3
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 5: N2O
    species[5] = +4.14571488e+04 * invT2 - 6.35015062e+02 * invT +
                 5.02874401e+00 + 6.33511505e-04 * T + 2.50385176e-06 * T2 -
                 2.86988796e-09 * T3 + 9.20159671e-13 * T4;
    // species 6: NO2
    species[6] = -5.64203878e+04 * invT2 + 9.63308572e+02 * invT -
                 3.43451097e+00 + 1.92776089e-02 * T - 1.87455933e-05 * T2 +
                 9.14549773e-09 * T3 - 1.77764763e-12 * T4;
    // species 7: N2O3
    species[7] = -9.20444417e+04 * invT2 + 9.29552015e+02 * invT +
                 2.20366481e+00 + 1.35647308e-02 * T - 6.26296607e-06 * T2 -
                 1.40291556e-09 * T3 + 1.43162093e-12 * T4;
    // species 8: NO3
    species[8] = +3.40539841e+04 * invT2 + 2.26667065e+02 * invT -
                 4.79308163e+00 + 4.17073270e-02 * T - 5.70991327e-05 * T2 +
                 3.83415811e-08 * T3 - 1.02196928e-11 * T4;
  } else {
    // species 5: N2O
    species[5] = +3.01106523e+05 * invT2 - 2.23674828e+03 * invT +
                 8.02146657e+00 - 5.70730355e-04 * T + 1.20484146e-07 * T2 -
                 1.33614030e-11 * T3 + 6.03852425e-16 * T4;
    // species 6: NO2
    species[6] = +7.21300157e+05 * invT2 - 3.83261520e+03 * invT +
                 1.01396328e+01 - 2.23806225e-03 * T + 6.54772343e-07 * T2 -
                 7.61133590e-11 * T3 + 3.32836105e-15 * T4;
    // species 7: N2O3
    species[7] = +7.78388186e+05 * invT2 - 4.48302466e+03 * invT +
                 1.56666802e+01 - 2.06214388e-03 * T + 5.30954171e-07 * T2 -
                 6.19045122e-11 * T3 + 2.69295666e-15 * T4;
    // species 8: NO3
    species[8] = -3.94387271e+05 * invT2 - 8.24426353e+02 * invT +
                 9.61325843e+00 - 2.44874982e-04 * T + 5.40606032e-08 * T2 -
                 6.19546675e-12 * T3 + 2.87000015e-16 * T4;
  }

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = +2.21037150e+04 * invT2 - 3.81846182e+02 * invT +
                 5.08273836e+00 - 8.53091441e-03 * T + 1.38464619e-05 * T2 -
                 9.62579362e-09 * T3 + 2.51970581e-12 * T4;
    // species 1: N
    species[1] = +1.50000000e+00;
    // species 2: O
    species[2] = -7.95361130e+03 * invT2 + 1.60717779e+02 * invT +
                 9.66226438e-01 + 1.01367031e-03 * T - 1.11041542e-06 * T2 +
                 6.51750750e-10 * T3 - 1.58477925e-13 * T4;
    // species 3: O2
    species[3] = -3.42556342e+04 * invT2 + 4.84700097e+02 * invT +
                 1.19010961e-01 + 4.29388924e-03 * T - 6.83630052e-07 * T2 -
                 2.02337270e-09 * T3 + 1.03904002e-12 * T4;
    // species 4: NO
    species[4] = -1.14391650e+04 * invT2 + 1.53646759e+02 * invT +
                 2.43146873e+00 - 2.66859237e-03 * T + 8.48139912e-06 * T2 -
                 7.68511105e-09 * T3 + 2.38679765e-12 * T4;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    species[0] = +5.87712406e+05 * invT2 - 2.23924907e+03 * invT +
                 5.06694922e+00 - 6.13968550e-04 * T + 1.49180668e-07 * T2 -
                 1.92310549e-11 * T3 + 1.06195439e-15 * T4;
    // species 1: N
    species[1] = +8.87650138e+04 * invT2 - 1.07123150e+02 * invT +
                 1.36218829e+00 + 2.91672008e-04 * T - 1.72951510e-07 * T2 +
                 4.01265788e-11 * T3 - 2.67722757e-15 * T4;
    // species 2: O
    species[2] = +2.61902026e+05 * invT2 - 7.29872203e+02 * invT +
                 2.31717727e+00 - 4.28133436e-04 * T + 1.03610459e-07 * T2 -
                 9.43830433e-12 * T3 + 2.72503830e-16 * T4;
    // species 3: O2
    species[3] = -1.03793902e+06 * invT2 + 2.34483028e+03 * invT +
                 8.19732036e-01 + 1.26784758e-03 * T - 2.18806799e-07 * T2 +
                 2.05371957e-11 * T3 - 8.19346705e-16 * T4;
    // species 4: NO
    species[4] = +2.23901872e+05 * invT2 - 1.28965162e+03 * invT +
                 4.43393603e+00 - 3.65603490e-04 * T + 9.88096645e-08 * T2 -
                 1.41607686e-11 * T3 + 9.38018462e-16 * T4;
  } else {
    // species 0: N2
    species[0] = +8.31013916e+08 * invT2 - 6.42073354e+05 * invT +
                 2.01026464e+02 - 3.06509205e-02 * T + 2.48690333e-06 * T2 -
                 9.70595411e-11 * T3 + 1.43753888e-15 * T4;
    // species 1: N
    species[1] = +5.47518105e+08 * invT2 - 3.10757498e+05 * invT +
                 6.81678274e+01 - 6.84798813e-03 * T + 3.82757240e-07 * T2 -
                 1.09836771e-11 * T3 + 1.27798602e-16 * T4;
    // species 2: O
    species[2] = +1.77900426e+08 * invT2 - 1.08232826e+05 * invT +
                 2.71077836e+01 - 2.97523226e-03 * T + 1.85499753e-07 * T2 -
                 5.79623154e-12 * T3 + 7.19172016e-17 * T4;
    // species 3: O2
    species[3] = +4.97529430e+08 * invT2 - 2.86610687e+05 * invT +
                 6.59035225e+01 - 6.16995902e-03 * T + 3.01639603e-07 * T2 -
                 7.42141660e-12 * T3 + 7.27817577e-17 * T4;
    // species 4: NO
    species[4] = -9.57530354e+08 * invT2 + 5.91243448e+05 * invT -
                 1.39456683e+02 + 1.69433940e-02 * T - 1.00735110e-06 * T2 +
                 2.91258408e-11 * T3 - 3.29510935e-16 * T4;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 5: N2O
    species[5] = +4.14571488e+04 * invT2 - 6.35015062e+02 * invT +
                 6.02874401e+00 + 6.33511505e-04 * T + 2.50385176e-06 * T2 -
                 2.86988796e-09 * T3 + 9.20159671e-13 * T4;
    // species 6: NO2
    species[6] = -5.64203878e+04 * invT2 + 9.63308572e+02 * invT -
                 2.43451097e+00 + 1.92776089e-02 * T - 1.87455933e-05 * T2 +
                 9.14549773e-09 * T3 - 1.77764763e-12 * T4;
    // species 7: N2O3
    species[7] = -9.20444417e+04 * invT2 + 9.29552015e+02 * invT +
                 3.20366481e+00 + 1.35647308e-02 * T - 6.26296607e-06 * T2 -
                 1.40291556e-09 * T3 + 1.43162093e-12 * T4;
    // species 8: NO3
    species[8] = +3.40539841e+04 * invT2 + 2.26667065e+02 * invT -
                 3.79308163e+00 + 4.17073270e-02 * T - 5.70991327e-05 * T2 +
                 3.83415811e-08 * T3 - 1.02196928e-11 * T4;
  } else {
    // species 5: N2O
    species[5] = +3.01106523e+05 * invT2 - 2.23674828e+03 * invT +
                 9.02146657e+00 - 5.70730355e-04 * T + 1.20484146e-07 * T2 -
                 1.33614030e-11 * T3 + 6.03852425e-16 * T4;
    // species 6: NO2
    species[6] = +7.21300157e+05 * invT2 - 3.83261520e+03 * invT +
                 1.11396328e+01 - 2.23806225e-03 * T + 6.54772343e-07 * T2 -
                 7.61133590e-11 * T3 + 3.32836105e-15 * T4;
    // species 7: N2O3
    species[7] = +7.78388186e+05 * invT2 - 4.48302466e+03 * invT +
                 1.66666802e+01 - 2.06214388e-03 * T + 5.30954171e-07 * T2 -
                 6.19045122e-11 * T3 + 2.69295666e-15 * T4;
    // species 8: NO3
    species[8] = -3.94387271e+05 * invT2 - 8.24426353e+02 * invT +
                 1.06132584e+01 - 2.44874982e-04 * T + 5.40606032e-08 * T2 -
                 6.19546675e-12 * T3 + 2.87000015e-16 * T4;
  }

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = +2.21037150e+04 * invT2 - 3.81846182e+02 * invT +
                 6.08273836e+00 - 8.53091441e-03 * T + 1.38464619e-05 * T2 -
                 9.62579362e-09 * T3 + 2.51970581e-12 * T4;
    // species 1: N
    species[1] = +2.50000000e+00;
    // species 2: O
    species[2] = -7.95361130e+03 * invT2 + 1.60717779e+02 * invT +
                 1.96622644e+00 + 1.01367031e-03 * T - 1.11041542e-06 * T2 +
                 6.51750750e-10 * T3 - 1.58477925e-13 * T4;
    // species 3: O2
    species[3] = -3.42556342e+04 * invT2 + 4.84700097e+02 * invT +
                 1.11901096e+00 + 4.29388924e-03 * T - 6.83630052e-07 * T2 -
                 2.02337270e-09 * T3 + 1.03904002e-12 * T4;
    // species 4: NO
    species[4] = -1.14391650e+04 * invT2 + 1.53646759e+02 * invT +
                 3.43146873e+00 - 2.66859237e-03 * T + 8.48139912e-06 * T2 -
                 7.68511105e-09 * T3 + 2.38679765e-12 * T4;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    species[0] = +5.87712406e+05 * invT2 - 2.23924907e+03 * invT +
                 6.06694922e+00 - 6.13968550e-04 * T + 1.49180668e-07 * T2 -
                 1.92310549e-11 * T3 + 1.06195439e-15 * T4;
    // species 1: N
    species[1] = +8.87650138e+04 * invT2 - 1.07123150e+02 * invT +
                 2.36218829e+00 + 2.91672008e-04 * T - 1.72951510e-07 * T2 +
                 4.01265788e-11 * T3 - 2.67722757e-15 * T4;
    // species 2: O
    species[2] = +2.61902026e+05 * invT2 - 7.29872203e+02 * invT +
                 3.31717727e+00 - 4.28133436e-04 * T + 1.03610459e-07 * T2 -
                 9.43830433e-12 * T3 + 2.72503830e-16 * T4;
    // species 3: O2
    species[3] = -1.03793902e+06 * invT2 + 2.34483028e+03 * invT +
                 1.81973204e+00 + 1.26784758e-03 * T - 2.18806799e-07 * T2 +
                 2.05371957e-11 * T3 - 8.19346705e-16 * T4;
    // species 4: NO
    species[4] = +2.23901872e+05 * invT2 - 1.28965162e+03 * invT +
                 5.43393603e+00 - 3.65603490e-04 * T + 9.88096645e-08 * T2 -
                 1.41607686e-11 * T3 + 9.38018462e-16 * T4;
  } else {
    // species 0: N2
    species[0] = +8.31013916e+08 * invT2 - 6.42073354e+05 * invT +
                 2.02026464e+02 - 3.06509205e-02 * T + 2.48690333e-06 * T2 -
                 9.70595411e-11 * T3 + 1.43753888e-15 * T4;
    // species 1: N
    species[1] = +5.47518105e+08 * invT2 - 3.10757498e+05 * invT +
                 6.91678274e+01 - 6.84798813e-03 * T + 3.82757240e-07 * T2 -
                 1.09836771e-11 * T3 + 1.27798602e-16 * T4;
    // species 2: O
    species[2] = +1.77900426e+08 * invT2 - 1.08232826e+05 * invT +
                 2.81077836e+01 - 2.97523226e-03 * T + 1.85499753e-07 * T2 -
                 5.79623154e-12 * T3 + 7.19172016e-17 * T4;
    // species 3: O2
    species[3] = +4.97529430e+08 * invT2 - 2.86610687e+05 * invT +
                 6.69035225e+01 - 6.16995902e-03 * T + 3.01639603e-07 * T2 -
                 7.42141660e-12 * T3 + 7.27817577e-17 * T4;
    // species 4: NO
    species[4] = -9.57530354e+08 * invT2 + 5.91243448e+05 * invT -
                 1.38456683e+02 + 1.69433940e-02 * T - 1.00735110e-06 * T2 +
                 2.91258408e-11 * T3 - 3.29510935e-16 * T4;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 5: N2O
    species[5] = -2.072857440000000e+04 * invT2 + 1.114787116800000e+04 * invT -
                 6.350150619999999e+02 * logT * invT -
                 6.028744010000000e+00 * logT + 1.605024370000000e+01 -
                 3.167557525000000e-04 * T - 4.173086265000000e-07 * T2 +
                 2.391573296666667e-10 * T3 - 4.600798355000000e-14 * T4;
    // species 6: NO2
    species[6] = +2.821019390000000e+04 * invT2 - 5.846164649999999e+02 * invT +
                 9.633085720000000e+02 * logT * invT +
                 2.434510974000000e+00 * logT - 4.311302307400000e+01 -
                 9.638804430000000e-03 * T + 3.124265546666667e-06 * T2 -
                 7.621248108333333e-10 * T3 + 8.888238174999999e-14 * T4;
    // species 7: N2O3
    species[7] = +4.602222085000000e+04 * invT2 + 4.243174095000000e+03 * invT +
                 9.295520150000000e+02 * logT * invT -
                 3.203664810000000e+00 * logT - 1.524064472000000e+01 -
                 6.782365390000000e-03 * T + 1.043827678333333e-06 * T2 +
                 1.169096299166667e-10 * T3 - 7.158104650000001e-14 * T4;
    // species 8: NO3
    species[8] = -1.702699205000000e+04 * invT2 + 7.314779265200000e+03 * invT +
                 2.266670652000000e+02 * logT * invT +
                 3.793081630000000e+00 * logT - 4.652399876000000e+01 -
                 2.085366350000000e-02 * T + 9.516522116666667e-06 * T2 -
                 3.195131758333333e-09 * T3 + 5.109846420000000e-13 * T4;
  } else {
    // species 5: N2O
    species[5] = -1.505532615000000e+05 * invT2 + 1.874380220700000e+04 * invT -
                 2.236748283000000e+03 * logT * invT -
                 9.021466569999999e+00 * logT + 3.951423077000000e+01 +
                 2.853651775000000e-04 * T - 2.008069091666667e-08 * T2 +
                 1.113450251666667e-12 * T3 - 3.019262125000000e-17 * T4;
    // species 6: NO2
    species[6] = -3.606500785000000e+05 * invT2 + 2.119235883000000e+04 * invT -
                 3.832615200000000e+03 * logT * invT -
                 1.113963285000000e+01 * logT + 5.419093325000000e+01 +
                 1.119031123000000e-03 * T - 1.091287238333333e-07 * T2 +
                 6.342779916666667e-12 * T3 - 1.664180525000000e-16 * T4;
    // species 7: N2O3
    species[7] = -3.891940930000000e+05 * invT2 + 2.912609984000000e+04 * invT -
                 4.483024660000000e+03 * logT * invT -
                 1.666668024000000e+01 * logT + 8.405880412000000e+01 +
                 1.031071939000000e-03 * T - 8.849236183333332e-08 * T2 +
                 5.158709350000000e-12 * T3 - 1.346478329000000e-16 * T4;
    // species 8: NO3
    species[8] = +1.971936355000000e+05 * invT2 + 8.157585377000000e+03 * invT -
                 8.244263529999999e+02 * logT * invT -
                 1.061325843000000e+01 * logT + 4.505992440000000e+01 +
                 1.224374908000000e-04 * T - 9.010100533333334e-09 * T2 +
                 5.162888958333333e-13 * T3 - 1.435000074500000e-17 * T4;
  }

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
    // species 2: O
    species[2] = +3.976805650000000e+03 * invT2 + 2.856434214870000e+04 * invT +
                 1.607177787000000e+02 * logT * invT -
                 1.966226438000000e+00 * logT - 6.438015382000000e+00 -
                 5.068351550000000e-04 * T + 1.850692371666667e-07 * T2 -
                 5.431256250000000e-11 * T3 + 7.923896255000001e-15 * T4;
    // species 3: O2
    species[3] = +1.712781710000000e+04 * invT2 - 2.906754773000000e+03 * invT +
                 4.847000970000000e+02 * logT * invT -
                 1.119010961000000e+00 * logT - 1.737798373900000e+01 -
                 2.146944620000000e-03 * T + 1.139383420000000e-07 * T2 +
                 1.686143916666667e-10 * T3 - 5.195200090000000e-14 * T4;
    // species 4: NO
    species[4] = +5.719582515000000e+03 * invT2 + 9.251861169200001e+03 * invT +
                 1.536467592000000e+02 * logT * invT -
                 3.431468730000000e+00 * logT - 3.297256760000000e+00 +
                 1.334296184000000e-03 * T - 1.413566520000000e-06 * T2 +
                 6.404259208333333e-10 * T3 - 1.193398827500000e-13 * T4;
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
    // species 2: O
    species[2] = -1.309510131000000e+05 * invT2 + 3.319440839700000e+04 * invT -
                 7.298722030000000e+02 * logT * invT -
                 3.317177270000000e+00 * logT + 3.985135805000000e+00 +
                 2.140667180000000e-04 * T - 1.726840990000000e-08 * T2 +
                 7.865253608333333e-13 * T3 - 1.362519148500000e-17 * T4;
    // species 3: O2
    species[3] = +5.189695110000000e+05 * invT2 - 1.454527900800000e+04 * invT +
                 2.344830282000000e+03 * logT * invT -
                 1.819732036000000e+00 * logT - 1.556743302400000e+01 -
                 6.339237910000000e-04 * T + 3.646779980000000e-08 * T2 -
                 1.711432976666667e-12 * T3 + 4.096733525000000e-17 * T4;
    // species 4: NO
    species[4] = -1.119509358000000e+05 * invT2 + 1.621352493700000e+04 * invT -
                 1.289651623000000e+03 * logT * invT -
                 5.433936030000000e+00 * logT + 1.393560512000000e+01 +
                 1.828017450000000e-04 * T - 1.646827741666667e-08 * T2 +
                 1.180064046666667e-12 * T3 - 4.690092310000000e-17 * T4;
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
    // species 2: O
    species[2] = -8.895021320000000e+07 * invT2 + 7.808614373000000e+05 * invT -
                 1.082328257000000e+05 * logT * invT -
                 2.810778365000000e+01 * logT + 2.462805987500000e+02 +
                 1.487616131000000e-03 * T - 3.091662556666666e-08 * T2 +
                 4.830192950000000e-13 * T3 - 3.595860082000000e-18 * T4;
    // species 3: O2
    species[3] = -2.487647150000000e+08 * invT2 + 2.006943339600000e+06 * invT -
                 2.866106874000000e+05 * logT * invT -
                 6.690352249999999e+01 * logT + 6.199656835000000e+02 +
                 3.084979510000000e-03 * T - 5.027326711666667e-08 * T2 +
                 6.184513833333334e-13 * T3 - 3.639087885000000e-18 * T4;
    // species 4: NO
    species[4] = +4.787651770000000e+08 * invT2 - 4.086257792000000e+06 * invT +
                 5.912434480000000e+05 * logT * invT +
                 1.384566826000000e+02 * logT - 1.380537898600000e+03 -
                 8.471697015000000e-03 * T + 1.678918493333333e-07 * T2 -
                 2.427153396666667e-12 * T3 + 1.647554675000000e-17 * T4;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 5: N2O
    species[5] = -2.07285744e+04 * invT2 + 1.11478712e+04 * invT -
                 6.35015062e+02 * logT * invT - 6.02874401e+00 * logT +
                 1.50502437e+01 - 3.16755752e-04 * T - 4.17308626e-07 * T2 +
                 2.39157330e-10 * T3 - 4.60079835e-14 * T4;
    // species 6: NO2
    species[6] = +2.82101939e+04 * invT2 - 5.84616465e+02 * invT +
                 9.63308572e+02 * logT * invT + 2.43451097e+00 * logT -
                 4.41130231e+01 - 9.63880443e-03 * T + 3.12426555e-06 * T2 -
                 7.62124811e-10 * T3 + 8.88823817e-14 * T4;
    // species 7: N2O3
    species[7] = +4.60222208e+04 * invT2 + 4.24317410e+03 * invT +
                 9.29552015e+02 * logT * invT - 3.20366481e+00 * logT -
                 1.62406447e+01 - 6.78236539e-03 * T + 1.04382768e-06 * T2 +
                 1.16909630e-10 * T3 - 7.15810465e-14 * T4;
    // species 8: NO3
    species[8] = -1.70269921e+04 * invT2 + 7.31477927e+03 * invT +
                 2.26667065e+02 * logT * invT + 3.79308163e+00 * logT -
                 4.75239988e+01 - 2.08536635e-02 * T + 9.51652212e-06 * T2 -
                 3.19513176e-09 * T3 + 5.10984642e-13 * T4;
  } else {
    // species 5: N2O
    species[5] = -1.50553261e+05 * invT2 + 1.87438022e+04 * invT -
                 2.23674828e+03 * logT * invT - 9.02146657e+00 * logT +
                 3.85142308e+01 + 2.85365177e-04 * T - 2.00806909e-08 * T2 +
                 1.11345025e-12 * T3 - 3.01926212e-17 * T4;
    // species 6: NO2
    species[6] = -3.60650079e+05 * invT2 + 2.11923588e+04 * invT -
                 3.83261520e+03 * logT * invT - 1.11396328e+01 * logT +
                 5.31909333e+01 + 1.11903112e-03 * T - 1.09128724e-07 * T2 +
                 6.34277992e-12 * T3 - 1.66418053e-16 * T4;
    // species 7: N2O3
    species[7] = -3.89194093e+05 * invT2 + 2.91260998e+04 * invT -
                 4.48302466e+03 * logT * invT - 1.66666802e+01 * logT +
                 8.30588041e+01 + 1.03107194e-03 * T - 8.84923618e-08 * T2 +
                 5.15870935e-12 * T3 - 1.34647833e-16 * T4;
    // species 8: NO3
    species[8] = +1.97193636e+05 * invT2 + 8.15758538e+03 * invT -
                 8.24426353e+02 * logT * invT - 1.06132584e+01 * logT +
                 4.40599244e+01 + 1.22437491e-04 * T - 9.01010053e-09 * T2 +
                 5.16288896e-13 * T3 - 1.43500007e-17 * T4;
  }

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
    // species 2: O
    species[2] = +3.97680565e+03 * invT2 + 2.85643421e+04 * invT +
                 1.60717779e+02 * logT * invT - 1.96622644e+00 * logT -
                 7.43801538e+00 - 5.06835155e-04 * T + 1.85069237e-07 * T2 -
                 5.43125625e-11 * T3 + 7.92389626e-15 * T4;
    // species 3: O2
    species[3] = +1.71278171e+04 * invT2 - 2.90675477e+03 * invT +
                 4.84700097e+02 * logT * invT - 1.11901096e+00 * logT -
                 1.83779837e+01 - 2.14694462e-03 * T + 1.13938342e-07 * T2 +
                 1.68614392e-10 * T3 - 5.19520009e-14 * T4;
    // species 4: NO
    species[4] = +5.71958252e+03 * invT2 + 9.25186117e+03 * invT +
                 1.53646759e+02 * logT * invT - 3.43146873e+00 * logT -
                 4.29725676e+00 + 1.33429618e-03 * T - 1.41356652e-06 * T2 +
                 6.40425921e-10 * T3 - 1.19339883e-13 * T4;
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
    // species 2: O
    species[2] = -1.30951013e+05 * invT2 + 3.31944084e+04 * invT -
                 7.29872203e+02 * logT * invT - 3.31717727e+00 * logT +
                 2.98513581e+00 + 2.14066718e-04 * T - 1.72684099e-08 * T2 +
                 7.86525361e-13 * T3 - 1.36251915e-17 * T4;
    // species 3: O2
    species[3] = +5.18969511e+05 * invT2 - 1.45452790e+04 * invT +
                 2.34483028e+03 * logT * invT - 1.81973204e+00 * logT -
                 1.65674330e+01 - 6.33923791e-04 * T + 3.64677998e-08 * T2 -
                 1.71143298e-12 * T3 + 4.09673353e-17 * T4;
    // species 4: NO
    species[4] = -1.11950936e+05 * invT2 + 1.62135249e+04 * invT -
                 1.28965162e+03 * logT * invT - 5.43393603e+00 * logT +
                 1.29356051e+01 + 1.82801745e-04 * T - 1.64682774e-08 * T2 +
                 1.18006405e-12 * T3 - 4.69009231e-17 * T4;
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
    // species 2: O
    species[2] = -8.89502132e+07 * invT2 + 7.80861437e+05 * invT -
                 1.08232826e+05 * logT * invT - 2.81077836e+01 * logT +
                 2.45280599e+02 + 1.48761613e-03 * T - 3.09166256e-08 * T2 +
                 4.83019295e-13 * T3 - 3.59586008e-18 * T4;
    // species 3: O2
    species[3] = -2.48764715e+08 * invT2 + 2.00694334e+06 * invT -
                 2.86610687e+05 * logT * invT - 6.69035225e+01 * logT +
                 6.18965683e+02 + 3.08497951e-03 * T - 5.02732671e-08 * T2 +
                 6.18451383e-13 * T3 - 3.63908789e-18 * T4;
    // species 4: NO
    species[4] = +4.78765177e+08 * invT2 - 4.08625779e+06 * invT +
                 5.91243448e+05 * logT * invT + 1.38456683e+02 * logT -
                 1.38153790e+03 - 8.47169701e-03 * T + 1.67891849e-07 * T2 -
                 2.42715340e-12 * T3 + 1.64755467e-17 * T4;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 5: N2O
    species[5] = -4.14571488e+04 * invT2 - 6.35015062e+02 * logT * invT +
                 5.02874401e+00 + 3.16755752e-04 * T + 8.34617253e-07 * T2 -
                 7.17471989e-10 * T3 + 1.84031934e-13 * T4 +
                 1.17828862e+04 * invT;
    // species 6: NO2
    species[6] = +5.64203878e+04 * invT2 + 9.63308572e+02 * logT * invT -
                 3.43451097e+00 + 9.63880443e-03 * T - 6.24853109e-06 * T2 +
                 2.28637443e-09 * T3 - 3.55529527e-13 * T4 -
                 1.54792504e+03 * invT;
    // species 7: N2O3
    species[7] = +9.20444417e+04 * invT2 + 9.29552015e+02 * logT * invT +
                 2.20366481e+00 + 6.78236539e-03 * T - 2.08765536e-06 * T2 -
                 3.50728890e-10 * T3 + 2.86324186e-13 * T4 +
                 3.31362208e+03 * invT;
    // species 8: NO3
    species[8] = -3.40539841e+04 * invT2 + 2.26667065e+02 * logT * invT -
                 4.79308163e+00 + 2.08536635e-02 * T - 1.90330442e-05 * T2 +
                 9.58539527e-09 * T3 - 2.04393857e-12 * T4 +
                 7.08811220e+03 * invT;
  } else {
    // species 5: N2O
    species[5] = -3.01106523e+05 * invT2 - 2.23674828e+03 * logT * invT +
                 8.02146657e+00 - 2.85365177e-04 * T + 4.01613818e-08 * T2 -
                 3.34035075e-12 * T3 + 1.20770485e-16 * T4 +
                 2.09805505e+04 * invT;
    // species 6: NO2
    species[6] = -7.21300157e+05 * invT2 - 3.83261520e+03 * logT * invT +
                 1.01396328e+01 - 1.11903112e-03 * T + 2.18257448e-07 * T2 -
                 1.90283398e-11 * T3 + 6.65672210e-16 * T4 +
                 2.50249740e+04 * invT;
    // species 7: N2O3
    species[7] = -7.78388186e+05 * invT2 - 4.48302466e+03 * logT * invT +
                 1.56666802e+01 - 1.03107194e-03 * T + 1.76984724e-07 * T2 -
                 1.54761281e-11 * T3 + 5.38591332e-16 * T4 +
                 3.36091245e+04 * invT;
    // species 8: NO3
    species[8] = +3.94387271e+05 * invT2 - 8.24426353e+02 * logT * invT +
                 9.61325843e+00 - 1.22437491e-04 * T + 1.80202011e-08 * T2 -
                 1.54886669e-12 * T3 + 5.74000030e-17 * T4 +
                 8.98201173e+03 * invT;
  }

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = -2.21037150e+04 * invT2 - 3.81846182e+02 * logT * invT +
                 5.08273836e+00 - 4.26545720e-03 * T + 4.61548730e-06 * T2 -
                 2.40644840e-09 * T3 + 5.03941162e-13 * T4 +
                 7.10846086e+02 * invT;
    // species 1: N
    species[1] = +1.50000000e+00 + 5.61046378e+04 * invT;
    // species 2: O
    species[2] = +7.95361130e+03 * invT2 + 1.60717779e+02 * logT * invT +
                 9.66226438e-01 + 5.06835155e-04 * T - 3.70138474e-07 * T2 +
                 1.62937687e-10 * T3 - 3.16955850e-14 * T4 +
                 2.84036244e+04 * invT;
    // species 3: O2
    species[3] = +3.42556342e+04 * invT2 + 4.84700097e+02 * logT * invT +
                 1.19010961e-01 + 2.14694462e-03 * T - 2.27876684e-07 * T2 -
                 5.05843175e-10 * T3 + 2.07808004e-13 * T4 -
                 3.39145487e+03 * invT;
    // species 4: NO
    species[4] = +1.14391650e+04 * invT2 + 1.53646759e+02 * logT * invT +
                 2.43146873e+00 - 1.33429618e-03 * T + 2.82713304e-06 * T2 -
                 1.92127776e-09 * T3 + 4.77359531e-13 * T4 +
                 9.09821441e+03 * invT;
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
    // species 2: O
    species[2] = -2.61902026e+05 * invT2 - 7.29872203e+02 * logT * invT +
                 2.31717727e+00 - 2.14066718e-04 * T + 3.45368198e-08 * T2 -
                 2.35957608e-12 * T3 + 5.45007659e-17 * T4 +
                 3.39242806e+04 * invT;
    // species 3: O2
    species[3] = +1.03793902e+06 * invT2 + 2.34483028e+03 * logT * invT +
                 8.19732036e-01 + 6.33923791e-04 * T - 7.29355996e-08 * T2 +
                 5.13429893e-12 * T3 - 1.63869341e-16 * T4 -
                 1.68901093e+04 * invT;
    // species 4: NO
    species[4] = -2.23901872e+05 * invT2 - 1.28965162e+03 * logT * invT +
                 4.43393603e+00 - 1.82801745e-04 * T + 3.29365548e-08 * T2 -
                 3.54019214e-12 * T3 + 1.87603692e-16 * T4 +
                 1.75031766e+04 * invT;
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
    // species 2: O
    species[2] = -1.77900426e+08 * invT2 - 1.08232826e+05 * logT * invT +
                 2.71077836e+01 - 1.48761613e-03 * T + 6.18332511e-08 * T2 -
                 1.44905788e-12 * T3 + 1.43834403e-17 * T4 +
                 8.89094263e+05 * invT;
    // species 3: O2
    species[3] = -4.97529430e+08 * invT2 - 2.86610687e+05 * logT * invT +
                 6.59035225e+01 - 3.08497951e-03 * T + 1.00546534e-07 * T2 -
                 1.85535415e-12 * T3 + 1.45563515e-17 * T4 +
                 2.29355403e+06 * invT;
    // species 4: NO
    species[4] = +9.57530354e+08 * invT2 + 5.91243448e+05 * logT * invT -
                 1.39456683e+02 + 8.47169701e-03 * T - 3.35783699e-07 * T2 +
                 7.28146019e-12 * T3 - 6.59021870e-17 * T4 -
                 4.67750124e+06 * invT;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 5: N2O
    species[5] = -4.14571488e+04 * invT2 - 6.35015062e+02 * logT * invT +
                 6.02874401e+00 + 3.16755752e-04 * T + 8.34617253e-07 * T2 -
                 7.17471989e-10 * T3 + 1.84031934e-13 * T4 +
                 1.17828862e+04 * invT;
    // species 6: NO2
    species[6] = +5.64203878e+04 * invT2 + 9.63308572e+02 * logT * invT -
                 2.43451097e+00 + 9.63880443e-03 * T - 6.24853109e-06 * T2 +
                 2.28637443e-09 * T3 - 3.55529527e-13 * T4 -
                 1.54792504e+03 * invT;
    // species 7: N2O3
    species[7] = +9.20444417e+04 * invT2 + 9.29552015e+02 * logT * invT +
                 3.20366481e+00 + 6.78236539e-03 * T - 2.08765536e-06 * T2 -
                 3.50728890e-10 * T3 + 2.86324186e-13 * T4 +
                 3.31362208e+03 * invT;
    // species 8: NO3
    species[8] = -3.40539841e+04 * invT2 + 2.26667065e+02 * logT * invT -
                 3.79308163e+00 + 2.08536635e-02 * T - 1.90330442e-05 * T2 +
                 9.58539527e-09 * T3 - 2.04393857e-12 * T4 +
                 7.08811220e+03 * invT;
  } else {
    // species 5: N2O
    species[5] = -3.01106523e+05 * invT2 - 2.23674828e+03 * logT * invT +
                 9.02146657e+00 - 2.85365177e-04 * T + 4.01613818e-08 * T2 -
                 3.34035075e-12 * T3 + 1.20770485e-16 * T4 +
                 2.09805505e+04 * invT;
    // species 6: NO2
    species[6] = -7.21300157e+05 * invT2 - 3.83261520e+03 * logT * invT +
                 1.11396328e+01 - 1.11903112e-03 * T + 2.18257448e-07 * T2 -
                 1.90283398e-11 * T3 + 6.65672210e-16 * T4 +
                 2.50249740e+04 * invT;
    // species 7: N2O3
    species[7] = -7.78388186e+05 * invT2 - 4.48302466e+03 * logT * invT +
                 1.66666802e+01 - 1.03107194e-03 * T + 1.76984724e-07 * T2 -
                 1.54761281e-11 * T3 + 5.38591332e-16 * T4 +
                 3.36091245e+04 * invT;
    // species 8: NO3
    species[8] = +3.94387271e+05 * invT2 - 8.24426353e+02 * logT * invT +
                 1.06132584e+01 - 1.22437491e-04 * T + 1.80202011e-08 * T2 -
                 1.54886669e-12 * T3 + 5.74000030e-17 * T4 +
                 8.98201173e+03 * invT;
  }

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = -2.21037150e+04 * invT2 - 3.81846182e+02 * logT * invT +
                 6.08273836e+00 - 4.26545720e-03 * T + 4.61548730e-06 * T2 -
                 2.40644840e-09 * T3 + 5.03941162e-13 * T4 +
                 7.10846086e+02 * invT;
    // species 1: N
    species[1] = +2.50000000e+00 + 5.61046378e+04 * invT;
    // species 2: O
    species[2] = +7.95361130e+03 * invT2 + 1.60717779e+02 * logT * invT +
                 1.96622644e+00 + 5.06835155e-04 * T - 3.70138474e-07 * T2 +
                 1.62937687e-10 * T3 - 3.16955850e-14 * T4 +
                 2.84036244e+04 * invT;
    // species 3: O2
    species[3] = +3.42556342e+04 * invT2 + 4.84700097e+02 * logT * invT +
                 1.11901096e+00 + 2.14694462e-03 * T - 2.27876684e-07 * T2 -
                 5.05843175e-10 * T3 + 2.07808004e-13 * T4 -
                 3.39145487e+03 * invT;
    // species 4: NO
    species[4] = +1.14391650e+04 * invT2 + 1.53646759e+02 * logT * invT +
                 3.43146873e+00 - 1.33429618e-03 * T + 2.82713304e-06 * T2 -
                 1.92127776e-09 * T3 + 4.77359531e-13 * T4 +
                 9.09821441e+03 * invT;
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
    // species 2: O
    species[2] = -2.61902026e+05 * invT2 - 7.29872203e+02 * logT * invT +
                 3.31717727e+00 - 2.14066718e-04 * T + 3.45368198e-08 * T2 -
                 2.35957608e-12 * T3 + 5.45007659e-17 * T4 +
                 3.39242806e+04 * invT;
    // species 3: O2
    species[3] = +1.03793902e+06 * invT2 + 2.34483028e+03 * logT * invT +
                 1.81973204e+00 + 6.33923791e-04 * T - 7.29355996e-08 * T2 +
                 5.13429893e-12 * T3 - 1.63869341e-16 * T4 -
                 1.68901093e+04 * invT;
    // species 4: NO
    species[4] = -2.23901872e+05 * invT2 - 1.28965162e+03 * logT * invT +
                 5.43393603e+00 - 1.82801745e-04 * T + 3.29365548e-08 * T2 -
                 3.54019214e-12 * T3 + 1.87603692e-16 * T4 +
                 1.75031766e+04 * invT;
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
    // species 2: O
    species[2] = -1.77900426e+08 * invT2 - 1.08232826e+05 * logT * invT +
                 2.81077836e+01 - 1.48761613e-03 * T + 6.18332511e-08 * T2 -
                 1.44905788e-12 * T3 + 1.43834403e-17 * T4 +
                 8.89094263e+05 * invT;
    // species 3: O2
    species[3] = -4.97529430e+08 * invT2 - 2.86610687e+05 * logT * invT +
                 6.69035225e+01 - 3.08497951e-03 * T + 1.00546534e-07 * T2 -
                 1.85535415e-12 * T3 + 1.45563515e-17 * T4 +
                 2.29355403e+06 * invT;
    // species 4: NO
    species[4] = +9.57530354e+08 * invT2 + 5.91243448e+05 * logT * invT -
                 1.38456683e+02 + 8.47169701e-03 * T - 3.35783699e-07 * T2 +
                 7.28146019e-12 * T3 - 6.59021870e-17 * T4 -
                 4.67750124e+06 * invT;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 5: N2O
    species[5] = -2.07285744e+04 * invT2 + 6.35015062e+02 * invT +
                 6.02874401e+00 * logT + 6.33511505e-04 * T +
                 1.25192588e-06 * T2 - 9.56629319e-10 * T3 +
                 2.30039918e-13 * T4 - 1.00214997e+01;
    // species 6: NO2
    species[6] = +2.82101939e+04 * invT2 - 9.63308572e+02 * invT -
                 2.43451097e+00 * logT + 1.92776089e-02 * T -
                 9.37279664e-06 * T2 + 3.04849924e-09 * T3 -
                 4.44411909e-13 * T4 + 4.06785121e+01;
    // species 7: N2O3
    species[7] = +4.60222208e+04 * invT2 - 9.29552015e+02 * invT +
                 3.20366481e+00 * logT + 1.35647308e-02 * T -
                 3.13148303e-06 * T2 - 4.67638520e-10 * T3 +
                 3.57905233e-13 * T4 + 1.84443095e+01;
    // species 8: NO3
    species[8] = -1.70269921e+04 * invT2 - 2.26667065e+02 * invT -
                 3.79308163e+00 * logT + 4.17073270e-02 * T -
                 2.85495663e-05 * T2 + 1.27805270e-08 * T3 -
                 2.55492321e-12 * T4 + 4.27309171e+01;
  } else {
    // species 5: N2O
    species[5] = -1.50553261e+05 * invT2 + 2.23674828e+03 * invT +
                 9.02146657e+00 * logT - 5.70730355e-04 * T +
                 6.02420728e-08 * T2 - 4.45380101e-12 * T3 +
                 1.50963106e-16 * T4 - 3.04927642e+01;
    // species 6: NO2
    species[6] = -3.60650079e+05 * invT2 + 3.83261520e+03 * invT +
                 1.11396328e+01 * logT - 2.23806225e-03 * T +
                 3.27386172e-07 * T2 - 2.53711197e-11 * T3 +
                 8.32090263e-16 * T4 - 4.30513004e+01;
    // species 7: N2O3
    species[7] = -3.89194093e+05 * invT2 + 4.48302466e+03 * invT +
                 1.66666802e+01 * logT - 2.06214388e-03 * T +
                 2.65477085e-07 * T2 - 2.06348374e-11 * T3 +
                 6.73239164e-16 * T4 - 6.73921239e+01;
    // species 8: NO3
    species[8] = +1.97193636e+05 * invT2 + 8.24426353e+02 * invT +
                 1.06132584e+01 * logT - 2.44874982e-04 * T +
                 2.70303016e-08 * T2 - 2.06515558e-12 * T3 +
                 7.17500037e-17 * T4 - 3.44466660e+01;
  }

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = -1.10518575e+04 * invT2 + 3.81846182e+02 * invT +
                 6.08273836e+00 * logT - 8.53091441e-03 * T +
                 6.92323094e-06 * T2 - 3.20859787e-09 * T3 +
                 6.29926452e-13 * T4 - 1.07600374e+01;
    // species 1: N
    species[1] = +2.50000000e+00 * logT + 4.19390504e+00;
    // species 2: O
    species[2] = +3.97680565e+03 * invT2 - 1.60717779e+02 * invT +
                 1.96622644e+00 * logT + 1.01367031e-03 * T -
                 5.55207711e-07 * T2 + 2.17250250e-10 * T3 -
                 3.96194813e-14 * T4 + 8.40424182e+00;
    // species 3: O2
    species[3] = +1.71278171e+04 * invT2 - 4.84700097e+02 * invT +
                 1.11901096e+00 * logT + 4.29388924e-03 * T -
                 3.41815026e-07 * T2 - 6.74457567e-10 * T3 +
                 2.59760005e-13 * T4 + 1.84969947e+01;
    // species 4: NO
    species[4] = +5.71958252e+03 * invT2 - 1.53646759e+02 * invT +
                 3.43146873e+00 * logT - 2.66859237e-03 * T +
                 4.24069956e-06 * T2 - 2.56170368e-09 * T3 +
                 5.96699414e-13 * T4 + 6.72872549e+00;
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
    // species 2: O
    species[2] = -1.30951013e+05 * invT2 + 7.29872203e+02 * invT +
                 3.31717727e+00 * logT - 4.28133436e-04 * T +
                 5.18052297e-08 * T2 - 3.14610144e-12 * T3 +
                 6.81259574e-17 * T4 - 6.67958535e-01;
    // species 3: O2
    species[3] = +5.18969511e+05 * invT2 - 2.34483028e+03 * invT +
                 1.81973204e+00 * logT + 1.26784758e-03 * T -
                 1.09403399e-07 * T2 + 6.84573191e-12 * T3 -
                 2.04836676e-16 * T4 + 1.73871651e+01;
    // species 4: NO
    species[4] = -1.11950936e+05 * invT2 + 1.28965162e+03 * invT +
                 5.43393603e+00 * logT - 3.65603490e-04 * T +
                 4.94048322e-08 * T2 - 4.72025619e-12 * T3 +
                 2.34504615e-16 * T4 - 8.50166909e+00;
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
    // species 2: O
    species[2] = -8.89502132e+07 * invT2 + 1.08232826e+05 * invT +
                 2.81077836e+01 * logT - 2.97523226e-03 * T +
                 9.27498767e-08 * T2 - 1.93207718e-12 * T3 +
                 1.79793004e-17 * T4 - 2.18172815e+02;
    // species 3: O2
    species[3] = -2.48764715e+08 * invT2 + 2.86610687e+05 * invT +
                 6.69035225e+01 * logT - 6.16995902e-03 * T +
                 1.50819801e-07 * T2 - 2.47380553e-12 * T3 +
                 1.81954394e-17 * T4 - 5.53062161e+02;
    // species 4: NO
    species[4] = +4.78765177e+08 * invT2 - 5.91243448e+05 * invT -
                 1.38456683e+02 * logT + 1.69433940e-02 * T -
                 5.03675548e-07 * T2 + 9.70861359e-12 * T3 -
                 8.23777337e-17 * T4 + 1.24208122e+03;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 5: N2O
    species[5] = -8.29142976e+04 * invT3 + 6.35015062e+02 * invT2 +
                 6.33511505e-04 + 5.00770352e-06 * T - 8.60966387e-09 * T2 +
                 3.68063868e-12 * T3;
    // species 6: NO2
    species[6] = +1.12840776e+05 * invT3 - 9.63308572e+02 * invT2 +
                 1.92776089e-02 - 3.74911866e-05 * T + 2.74364932e-08 * T2 -
                 7.11059054e-12 * T3;
    // species 7: N2O3
    species[7] = +1.84088883e+05 * invT3 - 9.29552015e+02 * invT2 +
                 1.35647308e-02 - 1.25259321e-05 * T - 4.20874668e-09 * T2 +
                 5.72648372e-12 * T3;
    // species 8: NO3
    species[8] = -6.81079682e+04 * invT3 - 2.26667065e+02 * invT2 +
                 4.17073270e-02 - 1.14198265e-04 * T + 1.15024743e-07 * T2 -
                 4.08787714e-11 * T3;
  } else {
    // species 5: N2O
    species[5] = -6.02213046e+05 * invT3 + 2.23674828e+03 * invT2 -
                 5.70730355e-04 + 2.40968291e-07 * T - 4.00842091e-11 * T2 +
                 2.41540970e-15 * T3;
    // species 6: NO2
    species[6] = -1.44260031e+06 * invT3 + 3.83261520e+03 * invT2 -
                 2.23806225e-03 + 1.30954469e-06 * T - 2.28340077e-10 * T2 +
                 1.33134442e-14 * T3;
    // species 7: N2O3
    species[7] = -1.55677637e+06 * invT3 + 4.48302466e+03 * invT2 -
                 2.06214388e-03 + 1.06190834e-06 * T - 1.85713537e-10 * T2 +
                 1.07718266e-14 * T3;
    // species 8: NO3
    species[8] = +7.88774542e+05 * invT3 + 8.24426353e+02 * invT2 -
                 2.44874982e-04 + 1.08121206e-07 * T - 1.85864002e-11 * T2 +
                 1.14800006e-15 * T3;
  }

  // species with inflection points at T = (1000.0, 6000.0) kelvin
  if (T < 1000) {
    // species 0: N2
    species[0] = -4.42074299e+04 * invT3 + 3.81846182e+02 * invT2 -
                 8.53091441e-03 + 2.76929238e-05 * T - 2.88773809e-08 * T2 +
                 1.00788232e-11 * T3;
    // species 1: N
    species[1] = 0.0;
    // species 2: O
    species[2] = +1.59072226e+04 * invT3 - 1.60717779e+02 * invT2 +
                 1.01367031e-03 - 2.22083085e-06 * T + 1.95525225e-09 * T2 -
                 6.33911700e-13 * T3;
    // species 3: O2
    species[3] = +6.85112684e+04 * invT3 - 4.84700097e+02 * invT2 +
                 4.29388924e-03 - 1.36726010e-06 * T - 6.07011810e-09 * T2 +
                 4.15616007e-12 * T3;
    // species 4: NO
    species[4] = +2.28783301e+04 * invT3 - 1.53646759e+02 * invT2 -
                 2.66859237e-03 + 1.69627982e-05 * T - 2.30553331e-08 * T2 +
                 9.54719062e-12 * T3;
  } else if ((1000 <= T) && (T < 6000)) {
    // species 0: N2
    species[0] = -1.17542481e+06 * invT3 + 2.23924907e+03 * invT2 -
                 6.13968550e-04 + 2.98361336e-07 * T - 5.76931646e-11 * T2 +
                 4.24781754e-15 * T3;
    // species 1: N
    species[1] = -1.77530028e+05 * invT3 + 1.07123150e+02 * invT2 +
                 2.91672008e-04 - 3.45903020e-07 * T + 1.20379736e-10 * T2 -
                 1.07089103e-14 * T3;
    // species 2: O
    species[2] = -5.23804052e+05 * invT3 + 7.29872203e+02 * invT2 -
                 4.28133436e-04 + 2.07220919e-07 * T - 2.83149130e-11 * T2 +
                 1.09001532e-15 * T3;
    // species 3: O2
    species[3] = +2.07587804e+06 * invT3 - 2.34483028e+03 * invT2 +
                 1.26784758e-03 - 4.37613598e-07 * T + 6.16115872e-11 * T2 -
                 3.27738682e-15 * T3;
    // species 4: NO
    species[4] = -4.47803743e+05 * invT3 + 1.28965162e+03 * invT2 -
                 3.65603490e-04 + 1.97619329e-07 * T - 4.24823057e-11 * T2 +
                 3.75207385e-15 * T3;
  } else {
    // species 0: N2
    species[0] = -1.66202783e+09 * invT3 + 6.42073354e+05 * invT2 -
                 3.06509205e-02 + 4.97380667e-06 * T - 2.91178623e-10 * T2 +
                 5.75015552e-15 * T3;
    // species 1: N
    species[1] = -1.09503621e+09 * invT3 + 3.10757498e+05 * invT2 -
                 6.84798813e-03 + 7.65514480e-07 * T - 3.29510313e-11 * T2 +
                 5.11194410e-16 * T3;
    // species 2: O
    species[2] = -3.55800853e+08 * invT3 + 1.08232826e+05 * invT2 -
                 2.97523226e-03 + 3.70999507e-07 * T - 1.73886946e-11 * T2 +
                 2.87668807e-16 * T3;
    // species 3: O2
    species[3] = -9.95058860e+08 * invT3 + 2.86610687e+05 * invT2 -
                 6.16995902e-03 + 6.03279205e-07 * T - 2.22642498e-11 * T2 +
                 2.91127031e-16 * T3;
    // species 4: NO
    species[4] = +1.91506071e+09 * invT3 - 5.91243448e+05 * invT2 +
                 1.69433940e-02 - 2.01470219e-06 * T + 8.73775223e-11 * T2 -
                 1.31804374e-15 * T3;
  }
}

// Returns the mean specific heat at CP (Eq. 33)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cpbl)
{
  amrex::Real result = 0;
  amrex::Real cpor[9]; //  temporary storage
  cp_R(cpor, T);

  // perform dot product
  for (int id = 0; id < 9; ++id) {
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 5: N2O
    result += y[5] *
              (+4.14571488e+04 * invT2 - 6.35015062e+02 * invT +
               6.02874401e+00 + 6.33511505e-04 * T + 2.50385176e-06 * T2 -
               2.86988796e-09 * T3 + 9.20159671e-13 * T4) *
              0.0227205598345943;
    // species 6: NO2
    result += y[6] *
              (-5.64203878e+04 * invT2 + 9.63308572e+02 * invT -
               2.43451097e+00 + 1.92776089e-02 * T - 1.87455933e-05 * T2 +
               9.14549773e-09 * T3 - 1.77764763e-12 * T4) *
              0.0217367677426367;
    // species 7: N2O3
    result += y[7] *
              (-9.20444417e+04 * invT2 + 9.29552015e+02 * invT +
               3.20366481e+00 + 1.35647308e-02 * T - 6.26296607e-06 * T2 -
               1.40291556e-09 * T3 + 1.43162093e-12 * T4) *
              0.0131559905803107;
    // species 8: NO3
    result += y[8] *
              (+3.40539841e+04 * invT2 + 2.26667065e+02 * invT -
               3.79308163e+00 + 4.17073270e-02 * T - 5.70991327e-05 * T2 +
               3.83415811e-08 * T3 - 1.02196928e-11 * T4) *
              0.0161279917424682;
  } else {
    // species 5: N2O
    result += y[5] *
              (+3.01106523e+05 * invT2 - 2.23674828e+03 * invT +
               9.02146657e+00 - 5.70730355e-04 * T + 1.20484146e-07 * T2 -
               1.33614030e-11 * T3 + 6.03852425e-16 * T4) *
              0.0227205598345943;
    // species 6: NO2
    result += y[6] *
              (+7.21300157e+05 * invT2 - 3.83261520e+03 * invT +
               1.11396328e+01 - 2.23806225e-03 * T + 6.54772343e-07 * T2 -
               7.61133590e-11 * T3 + 3.32836105e-15 * T4) *
              0.0217367677426367;
    // species 7: N2O3
    result += y[7] *
              (+7.78388186e+05 * invT2 - 4.48302466e+03 * invT +
               1.66666802e+01 - 2.06214388e-03 * T + 5.30954171e-07 * T2 -
               6.19045122e-11 * T3 + 2.69295666e-15 * T4) *
              0.0131559905803107;
    // species 8: NO3
    result += y[8] *
              (-3.94387271e+05 * invT2 - 8.24426353e+02 * invT +
               1.06132584e+01 - 2.44874982e-04 * T + 5.40606032e-08 * T2 -
               6.19546675e-12 * T3 + 2.87000015e-16 * T4) *
              0.0161279917424682;
  }

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
    // species 2: O
    result += y[2] *
              (-7.95361130e+03 * invT2 + 1.60717779e+02 * invT +
               1.96622644e+00 + 1.01367031e-03 * T - 1.11041542e-06 * T2 +
               6.51750750e-10 * T3 - 1.58477925e-13 * T4) *
              0.0625039064941559;
    // species 3: O2
    result += y[3] *
              (-3.42556342e+04 * invT2 + 4.84700097e+02 * invT +
               1.11901096e+00 + 4.29388924e-03 * T - 6.83630052e-07 * T2 -
               2.02337270e-09 * T3 + 1.03904002e-12 * T4) *
              0.0312519532470779;
    // species 4: NO
    result += y[4] *
              (-1.14391650e+04 * invT2 + 1.53646759e+02 * invT +
               3.43146873e+00 - 2.66859237e-03 * T + 8.48139912e-06 * T2 -
               7.68511105e-09 * T3 + 2.38679765e-12 * T4) *
              0.0333266679997334;
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
    // species 2: O
    result += y[2] *
              (+2.61902026e+05 * invT2 - 7.29872203e+02 * invT +
               3.31717727e+00 - 4.28133436e-04 * T + 1.03610459e-07 * T2 -
               9.43830433e-12 * T3 + 2.72503830e-16 * T4) *
              0.0625039064941559;
    // species 3: O2
    result += y[3] *
              (-1.03793902e+06 * invT2 + 2.34483028e+03 * invT +
               1.81973204e+00 + 1.26784758e-03 * T - 2.18806799e-07 * T2 +
               2.05371957e-11 * T3 - 8.19346705e-16 * T4) *
              0.0312519532470779;
    // species 4: NO
    result += y[4] *
              (+2.23901872e+05 * invT2 - 1.28965162e+03 * invT +
               5.43393603e+00 - 3.65603490e-04 * T + 9.88096645e-08 * T2 -
               1.41607686e-11 * T3 + 9.38018462e-16 * T4) *
              0.0333266679997334;
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
    // species 2: O
    result += y[2] *
              (+1.77900426e+08 * invT2 - 1.08232826e+05 * invT +
               2.81077836e+01 - 2.97523226e-03 * T + 1.85499753e-07 * T2 -
               5.79623154e-12 * T3 + 7.19172016e-17 * T4) *
              0.0625039064941559;
    // species 3: O2
    result += y[3] *
              (+4.97529430e+08 * invT2 - 2.86610687e+05 * invT +
               6.69035225e+01 - 6.16995902e-03 * T + 3.01639603e-07 * T2 -
               7.42141660e-12 * T3 + 7.27817577e-17 * T4) *
              0.0312519532470779;
    // species 4: NO
    result += y[4] *
              (-9.57530354e+08 * invT2 + 5.91243448e+05 * invT -
               1.38456683e+02 + 1.69433940e-02 * T - 1.00735110e-06 * T2 +
               2.91258408e-11 * T3 - 3.29510935e-16 * T4) *
              0.0333266679997334;
  }

  cpbs = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 35)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cvbl)
{
  amrex::Real result = 0;
  amrex::Real cvor[9]; //  temporary storage
  cv_R(cvor, T);

  // perform dot product
  for (int id = 0; id < 9; ++id) {
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 5: N2O
    result += y[5] *
              (+4.14571488e+04 * invT2 - 6.35015062e+02 * invT +
               5.02874401e+00 + 6.33511505e-04 * T + 2.50385176e-06 * T2 -
               2.86988796e-09 * T3 + 9.20159671e-13 * T4) *
              0.0227205598345943;
    // species 6: NO2
    result += y[6] *
              (-5.64203878e+04 * invT2 + 9.63308572e+02 * invT -
               3.43451097e+00 + 1.92776089e-02 * T - 1.87455933e-05 * T2 +
               9.14549773e-09 * T3 - 1.77764763e-12 * T4) *
              0.0217367677426367;
    // species 7: N2O3
    result += y[7] *
              (-9.20444417e+04 * invT2 + 9.29552015e+02 * invT +
               2.20366481e+00 + 1.35647308e-02 * T - 6.26296607e-06 * T2 -
               1.40291556e-09 * T3 + 1.43162093e-12 * T4) *
              0.0131559905803107;
    // species 8: NO3
    result += y[8] *
              (+3.40539841e+04 * invT2 + 2.26667065e+02 * invT -
               4.79308163e+00 + 4.17073270e-02 * T - 5.70991327e-05 * T2 +
               3.83415811e-08 * T3 - 1.02196928e-11 * T4) *
              0.0161279917424682;
  } else {
    // species 5: N2O
    result += y[5] *
              (+3.01106523e+05 * invT2 - 2.23674828e+03 * invT +
               8.02146657e+00 - 5.70730355e-04 * T + 1.20484146e-07 * T2 -
               1.33614030e-11 * T3 + 6.03852425e-16 * T4) *
              0.0227205598345943;
    // species 6: NO2
    result += y[6] *
              (+7.21300157e+05 * invT2 - 3.83261520e+03 * invT +
               1.01396328e+01 - 2.23806225e-03 * T + 6.54772343e-07 * T2 -
               7.61133590e-11 * T3 + 3.32836105e-15 * T4) *
              0.0217367677426367;
    // species 7: N2O3
    result += y[7] *
              (+7.78388186e+05 * invT2 - 4.48302466e+03 * invT +
               1.56666802e+01 - 2.06214388e-03 * T + 5.30954171e-07 * T2 -
               6.19045122e-11 * T3 + 2.69295666e-15 * T4) *
              0.0131559905803107;
    // species 8: NO3
    result += y[8] *
              (-3.94387271e+05 * invT2 - 8.24426353e+02 * invT +
               9.61325843e+00 - 2.44874982e-04 * T + 5.40606032e-08 * T2 -
               6.19546675e-12 * T3 + 2.87000015e-16 * T4) *
              0.0161279917424682;
  }

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
    // species 2: O
    result += y[2] *
              (-7.95361130e+03 * invT2 + 1.60717779e+02 * invT +
               9.66226438e-01 + 1.01367031e-03 * T - 1.11041542e-06 * T2 +
               6.51750750e-10 * T3 - 1.58477925e-13 * T4) *
              0.0625039064941559;
    // species 3: O2
    result += y[3] *
              (-3.42556342e+04 * invT2 + 4.84700097e+02 * invT +
               1.19010961e-01 + 4.29388924e-03 * T - 6.83630052e-07 * T2 -
               2.02337270e-09 * T3 + 1.03904002e-12 * T4) *
              0.0312519532470779;
    // species 4: NO
    result += y[4] *
              (-1.14391650e+04 * invT2 + 1.53646759e+02 * invT +
               2.43146873e+00 - 2.66859237e-03 * T + 8.48139912e-06 * T2 -
               7.68511105e-09 * T3 + 2.38679765e-12 * T4) *
              0.0333266679997334;
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
    // species 2: O
    result += y[2] *
              (+2.61902026e+05 * invT2 - 7.29872203e+02 * invT +
               2.31717727e+00 - 4.28133436e-04 * T + 1.03610459e-07 * T2 -
               9.43830433e-12 * T3 + 2.72503830e-16 * T4) *
              0.0625039064941559;
    // species 3: O2
    result += y[3] *
              (-1.03793902e+06 * invT2 + 2.34483028e+03 * invT +
               8.19732036e-01 + 1.26784758e-03 * T - 2.18806799e-07 * T2 +
               2.05371957e-11 * T3 - 8.19346705e-16 * T4) *
              0.0312519532470779;
    // species 4: NO
    result += y[4] *
              (+2.23901872e+05 * invT2 - 1.28965162e+03 * invT +
               4.43393603e+00 - 3.65603490e-04 * T + 9.88096645e-08 * T2 -
               1.41607686e-11 * T3 + 9.38018462e-16 * T4) *
              0.0333266679997334;
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
    // species 2: O
    result += y[2] *
              (+1.77900426e+08 * invT2 - 1.08232826e+05 * invT +
               2.71077836e+01 - 2.97523226e-03 * T + 1.85499753e-07 * T2 -
               5.79623154e-12 * T3 + 7.19172016e-17 * T4) *
              0.0625039064941559;
    // species 3: O2
    result += y[3] *
              (+4.97529430e+08 * invT2 - 2.86610687e+05 * invT +
               6.59035225e+01 - 6.16995902e-03 * T + 3.01639603e-07 * T2 -
               7.42141660e-12 * T3 + 7.27817577e-17 * T4) *
              0.0312519532470779;
    // species 4: NO
    result += y[4] *
              (-9.57530354e+08 * invT2 + 5.91243448e+05 * invT -
               1.39456683e+02 + 1.69433940e-02 * T - 1.00735110e-06 * T2 +
               2.91258408e-11 * T3 - 3.29510935e-16 * T4) *
              0.0333266679997334;
  }

  cvbs = result * 8.31446261815324e+07;
}

// Returns the mean enthalpy of the mixture in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBML(const amrex::Real T, const amrex::Real x[], amrex::Real& hbml)
{
  amrex::Real result = 0;
  amrex::Real hml[9];                        //  temporary storage
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // perform dot product
  for (int id = 0; id < 9; ++id) {
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 5: N2O
    result +=
      y[5] *
      (-4.14571488e+04 * invT2 - 6.35015062e+02 * logT * invT + 6.02874401e+00 +
       3.16755752e-04 * T + 8.34617253e-07 * T2 - 7.17471989e-10 * T3 +
       1.84031934e-13 * T4 + 1.17828862e+04 * invT) *
      0.0227205598345943;
    // species 6: NO2
    result +=
      y[6] *
      (+5.64203878e+04 * invT2 + 9.63308572e+02 * logT * invT - 2.43451097e+00 +
       9.63880443e-03 * T - 6.24853109e-06 * T2 + 2.28637443e-09 * T3 -
       3.55529527e-13 * T4 - 1.54792504e+03 * invT) *
      0.0217367677426367;
    // species 7: N2O3
    result +=
      y[7] *
      (+9.20444417e+04 * invT2 + 9.29552015e+02 * logT * invT + 3.20366481e+00 +
       6.78236539e-03 * T - 2.08765536e-06 * T2 - 3.50728890e-10 * T3 +
       2.86324186e-13 * T4 + 3.31362208e+03 * invT) *
      0.0131559905803107;
    // species 8: NO3
    result +=
      y[8] *
      (-3.40539841e+04 * invT2 + 2.26667065e+02 * logT * invT - 3.79308163e+00 +
       2.08536635e-02 * T - 1.90330442e-05 * T2 + 9.58539527e-09 * T3 -
       2.04393857e-12 * T4 + 7.08811220e+03 * invT) *
      0.0161279917424682;
  } else {
    // species 5: N2O
    result +=
      y[5] *
      (-3.01106523e+05 * invT2 - 2.23674828e+03 * logT * invT + 9.02146657e+00 -
       2.85365177e-04 * T + 4.01613818e-08 * T2 - 3.34035075e-12 * T3 +
       1.20770485e-16 * T4 + 2.09805505e+04 * invT) *
      0.0227205598345943;
    // species 6: NO2
    result +=
      y[6] *
      (-7.21300157e+05 * invT2 - 3.83261520e+03 * logT * invT + 1.11396328e+01 -
       1.11903112e-03 * T + 2.18257448e-07 * T2 - 1.90283398e-11 * T3 +
       6.65672210e-16 * T4 + 2.50249740e+04 * invT) *
      0.0217367677426367;
    // species 7: N2O3
    result +=
      y[7] *
      (-7.78388186e+05 * invT2 - 4.48302466e+03 * logT * invT + 1.66666802e+01 -
       1.03107194e-03 * T + 1.76984724e-07 * T2 - 1.54761281e-11 * T3 +
       5.38591332e-16 * T4 + 3.36091245e+04 * invT) *
      0.0131559905803107;
    // species 8: NO3
    result +=
      y[8] *
      (+3.94387271e+05 * invT2 - 8.24426353e+02 * logT * invT + 1.06132584e+01 -
       1.22437491e-04 * T + 1.80202011e-08 * T2 - 1.54886669e-12 * T3 +
       5.74000030e-17 * T4 + 8.98201173e+03 * invT) *
      0.0161279917424682;
  }

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
    // species 2: O
    result +=
      y[2] *
      (+7.95361130e+03 * invT2 + 1.60717779e+02 * logT * invT + 1.96622644e+00 +
       5.06835155e-04 * T - 3.70138474e-07 * T2 + 1.62937687e-10 * T3 -
       3.16955850e-14 * T4 + 2.84036244e+04 * invT) *
      0.0625039064941559;
    // species 3: O2
    result +=
      y[3] *
      (+3.42556342e+04 * invT2 + 4.84700097e+02 * logT * invT + 1.11901096e+00 +
       2.14694462e-03 * T - 2.27876684e-07 * T2 - 5.05843175e-10 * T3 +
       2.07808004e-13 * T4 - 3.39145487e+03 * invT) *
      0.0312519532470779;
    // species 4: NO
    result +=
      y[4] *
      (+1.14391650e+04 * invT2 + 1.53646759e+02 * logT * invT + 3.43146873e+00 -
       1.33429618e-03 * T + 2.82713304e-06 * T2 - 1.92127776e-09 * T3 +
       4.77359531e-13 * T4 + 9.09821441e+03 * invT) *
      0.0333266679997334;
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
    // species 2: O
    result +=
      y[2] *
      (-2.61902026e+05 * invT2 - 7.29872203e+02 * logT * invT + 3.31717727e+00 -
       2.14066718e-04 * T + 3.45368198e-08 * T2 - 2.35957608e-12 * T3 +
       5.45007659e-17 * T4 + 3.39242806e+04 * invT) *
      0.0625039064941559;
    // species 3: O2
    result +=
      y[3] *
      (+1.03793902e+06 * invT2 + 2.34483028e+03 * logT * invT + 1.81973204e+00 +
       6.33923791e-04 * T - 7.29355996e-08 * T2 + 5.13429893e-12 * T3 -
       1.63869341e-16 * T4 - 1.68901093e+04 * invT) *
      0.0312519532470779;
    // species 4: NO
    result +=
      y[4] *
      (-2.23901872e+05 * invT2 - 1.28965162e+03 * logT * invT + 5.43393603e+00 -
       1.82801745e-04 * T + 3.29365548e-08 * T2 - 3.54019214e-12 * T3 +
       1.87603692e-16 * T4 + 1.75031766e+04 * invT) *
      0.0333266679997334;
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
    // species 2: O
    result +=
      y[2] *
      (-1.77900426e+08 * invT2 - 1.08232826e+05 * logT * invT + 2.81077836e+01 -
       1.48761613e-03 * T + 6.18332511e-08 * T2 - 1.44905788e-12 * T3 +
       1.43834403e-17 * T4 + 8.89094263e+05 * invT) *
      0.0625039064941559;
    // species 3: O2
    result +=
      y[3] *
      (-4.97529430e+08 * invT2 - 2.86610687e+05 * logT * invT + 6.69035225e+01 -
       3.08497951e-03 * T + 1.00546534e-07 * T2 - 1.85535415e-12 * T3 +
       1.45563515e-17 * T4 + 2.29355403e+06 * invT) *
      0.0312519532470779;
    // species 4: NO
    result +=
      y[4] *
      (+9.57530354e+08 * invT2 + 5.91243448e+05 * logT * invT - 1.38456683e+02 +
       8.47169701e-03 * T - 3.35783699e-07 * T2 + 7.28146019e-12 * T3 -
       6.59021870e-17 * T4 - 4.67750124e+06 * invT) *
      0.0333266679997334;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  hbms = result * RT;
}

// get mean internal energy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBML(const amrex::Real T, const amrex::Real x[], amrex::Real& ubml)
{
  amrex::Real result = 0;
  amrex::Real uml[9];                        //  temporary energy array
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // perform dot product
  for (int id = 0; id < 9; ++id) {
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 5: N2O
    result +=
      y[5] *
      (-4.14571488e+04 * invT2 - 6.35015062e+02 * logT * invT + 5.02874401e+00 +
       3.16755752e-04 * T + 8.34617253e-07 * T2 - 7.17471989e-10 * T3 +
       1.84031934e-13 * T4 + 1.17828862e+04 * invT) *
      0.0227205598345943;
    // species 6: NO2
    result +=
      y[6] *
      (+5.64203878e+04 * invT2 + 9.63308572e+02 * logT * invT - 3.43451097e+00 +
       9.63880443e-03 * T - 6.24853109e-06 * T2 + 2.28637443e-09 * T3 -
       3.55529527e-13 * T4 - 1.54792504e+03 * invT) *
      0.0217367677426367;
    // species 7: N2O3
    result +=
      y[7] *
      (+9.20444417e+04 * invT2 + 9.29552015e+02 * logT * invT + 2.20366481e+00 +
       6.78236539e-03 * T - 2.08765536e-06 * T2 - 3.50728890e-10 * T3 +
       2.86324186e-13 * T4 + 3.31362208e+03 * invT) *
      0.0131559905803107;
    // species 8: NO3
    result +=
      y[8] *
      (-3.40539841e+04 * invT2 + 2.26667065e+02 * logT * invT - 4.79308163e+00 +
       2.08536635e-02 * T - 1.90330442e-05 * T2 + 9.58539527e-09 * T3 -
       2.04393857e-12 * T4 + 7.08811220e+03 * invT) *
      0.0161279917424682;
  } else {
    // species 5: N2O
    result +=
      y[5] *
      (-3.01106523e+05 * invT2 - 2.23674828e+03 * logT * invT + 8.02146657e+00 -
       2.85365177e-04 * T + 4.01613818e-08 * T2 - 3.34035075e-12 * T3 +
       1.20770485e-16 * T4 + 2.09805505e+04 * invT) *
      0.0227205598345943;
    // species 6: NO2
    result +=
      y[6] *
      (-7.21300157e+05 * invT2 - 3.83261520e+03 * logT * invT + 1.01396328e+01 -
       1.11903112e-03 * T + 2.18257448e-07 * T2 - 1.90283398e-11 * T3 +
       6.65672210e-16 * T4 + 2.50249740e+04 * invT) *
      0.0217367677426367;
    // species 7: N2O3
    result +=
      y[7] *
      (-7.78388186e+05 * invT2 - 4.48302466e+03 * logT * invT + 1.56666802e+01 -
       1.03107194e-03 * T + 1.76984724e-07 * T2 - 1.54761281e-11 * T3 +
       5.38591332e-16 * T4 + 3.36091245e+04 * invT) *
      0.0131559905803107;
    // species 8: NO3
    result +=
      y[8] *
      (+3.94387271e+05 * invT2 - 8.24426353e+02 * logT * invT + 9.61325843e+00 -
       1.22437491e-04 * T + 1.80202011e-08 * T2 - 1.54886669e-12 * T3 +
       5.74000030e-17 * T4 + 8.98201173e+03 * invT) *
      0.0161279917424682;
  }

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
    // species 2: O
    result +=
      y[2] *
      (+7.95361130e+03 * invT2 + 1.60717779e+02 * logT * invT + 9.66226438e-01 +
       5.06835155e-04 * T - 3.70138474e-07 * T2 + 1.62937687e-10 * T3 -
       3.16955850e-14 * T4 + 2.84036244e+04 * invT) *
      0.0625039064941559;
    // species 3: O2
    result +=
      y[3] *
      (+3.42556342e+04 * invT2 + 4.84700097e+02 * logT * invT + 1.19010961e-01 +
       2.14694462e-03 * T - 2.27876684e-07 * T2 - 5.05843175e-10 * T3 +
       2.07808004e-13 * T4 - 3.39145487e+03 * invT) *
      0.0312519532470779;
    // species 4: NO
    result +=
      y[4] *
      (+1.14391650e+04 * invT2 + 1.53646759e+02 * logT * invT + 2.43146873e+00 -
       1.33429618e-03 * T + 2.82713304e-06 * T2 - 1.92127776e-09 * T3 +
       4.77359531e-13 * T4 + 9.09821441e+03 * invT) *
      0.0333266679997334;
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
    // species 2: O
    result +=
      y[2] *
      (-2.61902026e+05 * invT2 - 7.29872203e+02 * logT * invT + 2.31717727e+00 -
       2.14066718e-04 * T + 3.45368198e-08 * T2 - 2.35957608e-12 * T3 +
       5.45007659e-17 * T4 + 3.39242806e+04 * invT) *
      0.0625039064941559;
    // species 3: O2
    result +=
      y[3] *
      (+1.03793902e+06 * invT2 + 2.34483028e+03 * logT * invT + 8.19732036e-01 +
       6.33923791e-04 * T - 7.29355996e-08 * T2 + 5.13429893e-12 * T3 -
       1.63869341e-16 * T4 - 1.68901093e+04 * invT) *
      0.0312519532470779;
    // species 4: NO
    result +=
      y[4] *
      (-2.23901872e+05 * invT2 - 1.28965162e+03 * logT * invT + 4.43393603e+00 -
       1.82801745e-04 * T + 3.29365548e-08 * T2 - 3.54019214e-12 * T3 +
       1.87603692e-16 * T4 + 1.75031766e+04 * invT) *
      0.0333266679997334;
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
    // species 2: O
    result +=
      y[2] *
      (-1.77900426e+08 * invT2 - 1.08232826e+05 * logT * invT + 2.71077836e+01 -
       1.48761613e-03 * T + 6.18332511e-08 * T2 - 1.44905788e-12 * T3 +
       1.43834403e-17 * T4 + 8.89094263e+05 * invT) *
      0.0625039064941559;
    // species 3: O2
    result +=
      y[3] *
      (-4.97529430e+08 * invT2 - 2.86610687e+05 * logT * invT + 6.59035225e+01 -
       3.08497951e-03 * T + 1.00546534e-07 * T2 - 1.85535415e-12 * T3 +
       1.45563515e-17 * T4 + 2.29355403e+06 * invT) *
      0.0312519532470779;
    // species 4: NO
    result +=
      y[4] *
      (+9.57530354e+08 * invT2 + 5.91243448e+05 * logT * invT - 1.39456683e+02 +
       8.47169701e-03 * T - 3.35783699e-07 * T2 + 7.28146019e-12 * T3 -
       6.59021870e-17 * T4 - 4.67750124e+06 * invT) *
      0.0333266679997334;
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
  amrex::Real sor[9]; //  temporary storage
  speciesEntropy(sor, T);

  // Compute Eq 42
  for (int id = 0; id < 9; ++id) {
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
  amrex::Real sor[9];  //  temporary storage
  amrex::Real x[9];    //  need a ytx conversion
  amrex::Real YOW = 0; // See Eq 4, 6 in CK Manual

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }
  // Now compute y to x conversion
  x[0] = y[0] / (28.014000 * YOW);
  x[1] = y[1] / (14.007000 * YOW);
  x[2] = y[2] / (15.999000 * YOW);
  x[3] = y[3] / (31.998000 * YOW);
  x[4] = y[4] / (30.006000 * YOW);
  x[5] = y[5] / (44.013000 * YOW);
  x[6] = y[6] / (46.005000 * YOW);
  x[7] = y[7] / (76.011000 * YOW);
  x[8] = y[8] / (62.004000 * YOW);
  speciesEntropy(sor, T);
  // Perform computation in Eq 42 and 43
  for (int i = 0; i < 9; i++) {
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
  XW += x[2] * 15.999000;                  // O
  XW += x[3] * 31.998000;                  // O2
  XW += x[4] * 30.006000;                  // NO
  XW += x[5] * 44.013000;                  // N2O
  XW += x[6] * 46.005000;                  // NO2
  XW += x[7] * 76.011000;                  // N2O3
  XW += x[8] * 62.004000;                  // NO3
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

  for (int i = 0; i < 9; i++) {
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
  W += c[2] * 15.999000; // O
  W += c[3] * 31.998000; // O2
  W += c[4] * 30.006000; // NO
  W += c[5] * 44.013000; // N2O
  W += c[6] * 46.005000; // NO2
  W += c[7] * 76.011000; // N2O3
  W += c[8] * 62.004000; // NO3

  for (int id = 0; id < 9; ++id) {
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
  XW += x[2] * 15.999000;                    // O
  XW += x[3] * 31.998000;                    // O2
  XW += x[4] * 30.006000;                    // NO
  XW += x[5] * 44.013000;                    // N2O
  XW += x[6] * 46.005000;                    // NO2
  XW += x[7] * 76.011000;                    // N2O3
  XW += x[8] * 62.004000;                    // NO3
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

  for (int i = 0; i < 9; i++) {
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
  W += c[2] * 15.999000; // O
  W += c[3] * 31.998000; // O2
  W += c[4] * 30.006000; // NO
  W += c[5] * 44.013000; // N2O
  W += c[6] * 46.005000; // NO2
  W += c[7] * 76.011000; // N2O3
  W += c[8] * 62.004000; // NO3

  for (int id = 0; id < 9; ++id) {
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

  for (int i = 0; i < 9; i++) {
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
  XW += x[2] * 15.999000; // O
  XW += x[3] * 31.998000; // O2
  XW += x[4] * 30.006000; // NO
  XW += x[5] * 44.013000; // N2O
  XW += x[6] * 46.005000; // NO2
  XW += x[7] * 76.011000; // N2O3
  XW += x[8] * 62.004000; // NO3
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
  W += c[2] * 15.999000; // O
  W += c[3] * 31.998000; // O2
  W += c[4] * 30.006000; // NO
  W += c[5] * 44.013000; // N2O
  W += c[6] * 46.005000; // NO2
  W += c[7] * 76.011000; // N2O3
  W += c[8] * 62.004000; // NO3

  for (int id = 0; id < 9; ++id) {
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

  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }

  amrex::Real YOWINV = 1.0 / YOW;

  for (int i = 0; i < 9; i++) {
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
  for (int i = 0; i < 9; i++) {
    c[i] = y[i] * imw(i);
  }
  for (int i = 0; i < 9; i++) {
    YOW += c[i];
  }

  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // Now compute conversion

  for (int i = 0; i < 9; i++) {
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

  for (int i = 0; i < 9; i++) {
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
  XW += x[2] * 15.999000; // O
  XW += x[3] * 31.998000; // O2
  XW += x[4] * 30.006000; // NO
  XW += x[5] * 44.013000; // N2O
  XW += x[6] * 46.005000; // NO2
  XW += x[7] * 76.011000; // N2O3
  XW += x[8] * 62.004000; // NO3
  // Now compute conversion
  amrex::Real XWinv = 1.0 / XW;
  y[0] = x[0] * 28.014000 * XWinv;
  y[1] = x[1] * 14.007000 * XWinv;
  y[2] = x[2] * 15.999000 * XWinv;
  y[3] = x[3] * 31.998000 * XWinv;
  y[4] = x[4] * 30.006000 * XWinv;
  y[5] = x[5] * 44.013000 * XWinv;
  y[6] = x[6] * 46.005000 * XWinv;
  y[7] = x[7] * 76.011000 * XWinv;
  y[8] = x[8] * 62.004000 * XWinv;
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
  for (int id = 0; id < 9; ++id) {
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
  XW += x[2] * 15.999000; // O
  XW += x[3] * 31.998000; // O2
  XW += x[4] * 30.006000; // NO
  XW += x[5] * 44.013000; // N2O
  XW += x[6] * 46.005000; // NO2
  XW += x[7] * 76.011000; // N2O3
  XW += x[8] * 62.004000; // NO3
  ROW = rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 9; ++id) {
    c[id] = x[id] * ROW;
  }
}

// convert c[species] (molar conc) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTX(const amrex::Real c[], amrex::Real x[])
{
  amrex::Real sumC = 0;

  // compute sum of c
  for (int id = 0; id < 9; ++id) {
    sumC += c[id];
  }

  //  See Eq 13
  amrex::Real sumCinv = 1.0 / sumC;
  for (int id = 0; id < 9; ++id) {
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
  CW += c[2] * 15.999000; // O
  CW += c[3] * 31.998000; // O2
  CW += c[4] * 30.006000; // NO
  CW += c[5] * 44.013000; // N2O
  CW += c[6] * 46.005000; // NO2
  CW += c[7] * 76.011000; // N2O3
  CW += c[8] * 62.004000; // NO3
  // Now compute conversion
  amrex::Real CWinv = 1.0 / CW;
  y[0] = c[0] * 28.014000 * CWinv;
  y[1] = c[1] * 14.007000 * CWinv;
  y[2] = c[2] * 15.999000 * CWinv;
  y[3] = c[3] * 31.998000 * CWinv;
  y[4] = c[4] * 30.006000 * CWinv;
  y[5] = c[5] * 44.013000 * CWinv;
  y[6] = c[6] * 46.005000 * CWinv;
  y[7] = c[7] * 76.011000 * CWinv;
  y[8] = c[8] * 62.004000 * CWinv;
}

// get specific heat at constant volume as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVML(const amrex::Real T, amrex::Real cvml[])
{
  cv_R(cvml, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
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
  for (int id = 0; id < 9; ++id) {
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
  for (int id = 0; id < 9; ++id) {
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
  for (int id = 0; id < 9; ++id) {
    hml[id] *= RT;
  }
}

// Returns the standard-state entropies in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSML(const amrex::Real T, amrex::Real sml[])
{
  speciesEntropy(sml, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
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
  cvms[2] *= 5.196863940342046e+06; // O
  cvms[3] *= 2.598431970171023e+06; // O2
  cvms[4] *= 2.770933352713870e+06; // NO
  cvms[5] *= 1.889092454082485e+06; // N2O
  cvms[6] *= 1.807295428356318e+06; // NO2
  cvms[7] *= 1.093849918847698e+06; // N2O3
  cvms[8] *= 1.340955844486362e+06; // NO3
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
  cpms[2] *= 5.196863940342046e+06; // O
  cpms[3] *= 2.598431970171023e+06; // O2
  cpms[4] *= 2.770933352713870e+06; // NO
  cpms[5] *= 1.889092454082485e+06; // N2O
  cpms[6] *= 1.807295428356318e+06; // NO2
  cpms[7] *= 1.093849918847698e+06; // N2O3
  cpms[8] *= 1.340955844486362e+06; // NO3
}

// Returns internal energy in mass units (Eq 30.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUMS(const amrex::Real T, amrex::Real ums[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesInternalEnergy(ums, T);

  for (int i = 0; i < 9; i++) {
    ums[i] *= RT * imw(i);
  }
}

// Returns enthalpy in mass units (Eq 27.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHMS(const amrex::Real T, amrex::Real hms[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesEnthalpy(hms, T);

  for (int i = 0; i < 9; i++) {
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
  sms[2] *= 5.196863940342046e+06; // O
  sms[3] *= 2.598431970171023e+06; // O2
  sms[4] *= 2.770933352713870e+06; // NO
  sms[5] *= 1.889092454082485e+06; // N2O
  sms[6] *= 1.807295428356318e+06; // NO2
  sms[7] *= 1.093849918847698e+06; // N2O3
  sms[8] *= 1.340955844486362e+06; // NO3
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

  // reaction 2: O2 + M <=> 2 O + M
  qf[0] = sc[3];
  qr[0] = (sc[2] * sc[2]);

  // reaction 3: N2 + M <=> 2 N + M
  qf[1] = sc[0];
  qr[1] = (sc[1] * sc[1]);

  // reaction 4: NO + M <=> N + O + M
  qf[2] = sc[4];
  qr[2] = sc[1] * sc[2];

  // reaction 5: N2O + M <=> N2 + O + M
  qf[3] = sc[5];
  qr[3] = sc[0] * sc[2];

  // reaction 9: NO + O + M => NO2 + M
  qf[4] = sc[2] * sc[4];
  qr[4] = 0.0;

  // reaction 10: NO2 + M => NO + O + M
  qf[5] = sc[6];
  qr[5] = 0.0;

  // reaction 16: N2O3 + M => NO + NO2 + M
  qf[6] = sc[7];
  qr[6] = 0.0;

  // reaction 17: NO + NO2 + M => N2O3 + M
  qf[7] = sc[4] * sc[6];
  qr[7] = 0.0;

  // reaction 0: N2 + O <=> N + NO
  qf[8] = sc[0] * sc[2];
  qr[8] = sc[1] * sc[4];

  // reaction 1: NO + O <=> N + O2
  qf[9] = sc[2] * sc[4];
  qr[9] = sc[1] * sc[3];

  // reaction 6: N2O + O => 2 NO
  qf[10] = sc[2] * sc[5];
  qr[10] = 0.0;

  // reaction 7: 2 NO => N2O + O
  qf[11] = (sc[4] * sc[4]);
  qr[11] = 0.0;

  // reaction 8: N2O + O <=> N2 + O2
  qf[12] = sc[2] * sc[5];
  qr[12] = sc[0] * sc[3];

  // reaction 11: NO2 + O => NO + O2
  qf[13] = sc[2] * sc[6];
  qr[13] = 0.0;

  // reaction 12: 2 NO <=> N2 + O2
  qf[14] = (sc[4] * sc[4]);
  qr[14] = sc[0] * sc[3];

  // reaction 13: N2O + NO => N2 + NO2
  qf[15] = sc[4] * sc[5];
  qr[15] = 0.0;

  // reaction 14: 2 NO + O2 => 2 NO2
  qf[16] = sc[3] * (sc[4] * sc[4]);
  qr[16] = 0.0;

  // reaction 15: 2 NO2 => 2 NO + O2
  qf[17] = (sc[6] * sc[6]);
  qr[17] = 0.0;

  // reaction 18: 2 NO2 => NO + NO3
  qf[18] = (sc[6] * sc[6]);
  qr[18] = 0.0;

  // reaction 19: NO + NO3 => 2 NO2
  qf[19] = sc[4] * sc[8];
  qr[19] = 0.0;

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 9; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[9];
  gibbs(g_RT, T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 * invT;
  amrex::Real refCinv = 1 / refC;

  // Evaluate the kfs
  amrex::Real k_f, Corr;

  // reaction 0:  N2 + O <=> N + NO
  k_f = 181000000 * exp(-(38395.4316009001) * invT);
  qf[8] *= k_f;
  qr[8] *= k_f * exp(-(g_RT[0] - g_RT[1] + g_RT[2] - g_RT[4]));
  // reaction 1:  NO + O <=> N + O2
  k_f = 687 * exp((1.13) * logT - (19222.8766337403) * invT);
  qf[9] *= k_f;
  qr[9] *= k_f * exp(-(-g_RT[1] + g_RT[2] - g_RT[3] + g_RT[4]));
  // reaction 2:  O2 + M <=> 2 O + M
  k_f = 1170000000 * exp(-(60385.9998965664) * invT);
  Corr = mixture;
  qf[0] *= Corr * k_f;
  qr[0] *= Corr * k_f * exp(-(-2.000000 * g_RT[2] + g_RT[3])) * (refCinv);
  // reaction 3:  N2 + M <=> 2 N + M
  k_f = 6380000000 * exp(-(113726.966471867) * invT);
  Corr = mixture;
  qf[1] *= Corr * k_f;
  qr[1] *= Corr * k_f * exp(-(g_RT[0] - 2.000000 * g_RT[1])) * (refCinv);
  // reaction 4:  NO + M <=> N + O + M
  k_f = 1450000000 * exp(-(74476.0665390985) * invT);
  Corr = mixture;
  qf[2] *= Corr * k_f;
  qr[2] *= Corr * k_f * exp(-(-g_RT[1] - g_RT[2] + g_RT[4])) * (refCinv);
  // reaction 5:  N2O + M <=> N2 + O + M
  k_f = 723000000000 * exp((-0.73) * logT - (31602.0066125364) * invT);
  Corr = mixture;
  qf[3] *= Corr * k_f;
  qr[3] *= Corr * k_f * exp(-(-g_RT[0] - g_RT[2] + g_RT[5])) * (refCinv);
  // reaction 6:  N2O + O => 2 NO
  k_f = 90300000 * exp(-(13939.1016427907) * invT);
  qf[10] *= k_f;
  qr[10] *= k_f * exp(-(g_RT[2] - 2.000000 * g_RT[4] + g_RT[5]));
  // reaction 7:  2 NO => N2O + O
  k_f = 3610000 * exp(-(32910.3699436287) * invT);
  qf[11] *= k_f;
  qr[11] *= k_f * exp(-(-g_RT[2] + 2.000000 * g_RT[4] - g_RT[5]));
  // reaction 8:  N2O + O <=> N2 + O2
  k_f = 3670000 * exp(-(986304.664977251) * invT);
  qf[12] *= k_f;
  qr[12] *= k_f * exp(-(-g_RT[0] + g_RT[2] - g_RT[3] + g_RT[5]));
  // reaction 9:  NO + O + M => NO2 + M
  k_f = 4710000000000 * exp((-2.87) * logT);
  Corr = mixture;
  qf[4] *= Corr * k_f;
  qr[4] *= Corr * k_f * exp(-(g_RT[2] + g_RT[4] - g_RT[6])) * (refC);
  // reaction 10:  NO2 + M => NO + O + M
  k_f = 10800000000 * exp(-(33011.0132767896) * invT);
  Corr = mixture;
  qf[5] *= Corr * k_f;
  qr[5] *= Corr * k_f * exp(-(-g_RT[2] - g_RT[4] + g_RT[6])) * (refCinv);
  // reaction 11:  NO2 + O => NO + O2
  k_f = 3910000 * exp(-(-119.765566461523) * invT);
  qf[13] *= k_f;
  qr[13] *= k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[4] + g_RT[6]));
  // reaction 12:  2 NO <=> N2 + O2
  k_f = 12600000 * exp(-(30293.6432814441) * invT);
  qf[14] *= k_f;
  qr[14] *= k_f * exp(-(-g_RT[0] - g_RT[3] + 2.000000 * g_RT[4]));
  // reaction 13:  N2O + NO => N2 + NO2
  k_f = 18500000 * exp(-(24909.2249573336) * invT);
  qf[15] *= k_f;
  qr[15] *= k_f * exp(-(-g_RT[0] + g_RT[4] + g_RT[5] - g_RT[6]));
  // reaction 14:  2 NO + O2 => 2 NO2
  k_f = 0.0012 * exp(-(-528.377499094956) * invT);
  qf[16] *= k_f;
  qr[16] *=
    k_f * exp(-(g_RT[3] + 2.000000 * g_RT[4] - 2.000000 * g_RT[6])) * (refC);
  // reaction 15:  2 NO2 => 2 NO + O2
  k_f = 1630000 * exp(-(13385.5633104055) * invT);
  qf[17] *= k_f;
  qr[17] *= k_f * exp(-(-g_RT[3] - 2.000000 * g_RT[4] + 2.000000 * g_RT[6])) *
            (refCinv);
  // reaction 16:  N2O3 + M => NO + NO2 + M
  k_f = 65000000 * exp(-(2626.79099550064) * invT);
  Corr = mixture;
  qf[6] *= Corr * k_f;
  qr[6] *= Corr * k_f * exp(-(-g_RT[4] - g_RT[6] + g_RT[7])) * (refCinv);
  // reaction 17:  NO + NO2 + M => N2O3 + M
  k_f = 1.33e+21 * exp((-7.7) * logT);
  Corr = mixture;
  qf[7] *= Corr * k_f;
  qr[7] *= Corr * k_f * exp(-(g_RT[4] + g_RT[6] - g_RT[7])) * (refC);
  // reaction 18:  2 NO2 => NO + NO3
  k_f = 9640 * exp((0.73) * logT - (10517.2283153186) * invT);
  qf[18] *= k_f;
  qr[18] *= k_f * exp(-(-g_RT[4] + 2.000000 * g_RT[6] - g_RT[8]));
  // reaction 19:  NO + NO3 => 2 NO2
  k_f = 10800000 * exp(-(-108.191583148015) * invT);
  qf[19] *= k_f;
  qr[19] *= k_f * exp(-(g_RT[4] - 2.000000 * g_RT[6] + g_RT[8]));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
productionRate(amrex::Real* wdot, const amrex::Real* sc, const amrex::Real T)
{
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  const amrex::Real refC = 101325 / 8.31446 * invT;
  const amrex::Real refCinv = 1 / refC;

  for (int i = 0; i < 9; ++i) {
    wdot[i] = 0.0;
  }

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 9; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[9];
  gibbs(g_RT, T);

  {
    // reaction 2:  O2 + M <=> 2 O + M
    const amrex::Real k_f = 1170000000 * exp(-(60385.9998965664) * invT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[3]);
    const amrex::Real qr = Corr * k_f * exp(-(-2.000000 * g_RT[2] + g_RT[3])) *
                           (refCinv) * ((sc[2] * sc[2]));
    const amrex::Real qdot = qf - qr;
    wdot[2] += 2.000000 * qdot;
    wdot[3] -= qdot;
  }

  {
    // reaction 3:  N2 + M <=> 2 N + M
    const amrex::Real k_f = 6380000000 * exp(-(113726.966471867) * invT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[0]);
    const amrex::Real qr = Corr * k_f * exp(-(g_RT[0] - 2.000000 * g_RT[1])) *
                           (refCinv) * ((sc[1] * sc[1]));
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[1] += 2.000000 * qdot;
  }

  {
    // reaction 4:  NO + M <=> N + O + M
    const amrex::Real k_f = 1450000000 * exp(-(74476.0665390985) * invT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[4]);
    const amrex::Real qr = Corr * k_f * exp(-(-g_RT[1] - g_RT[2] + g_RT[4])) *
                           (refCinv) * (sc[1] * sc[2]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] += qdot;
    wdot[4] -= qdot;
  }

  {
    // reaction 5:  N2O + M <=> N2 + O + M
    const amrex::Real k_f =
      723000000000 * exp((-0.73) * logT - (31602.0066125364) * invT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[5]);
    const amrex::Real qr = Corr * k_f * exp(-(-g_RT[0] - g_RT[2] + g_RT[5])) *
                           (refCinv) * (sc[0] * sc[2]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[2] += qdot;
    wdot[5] -= qdot;
  }

  {
    // reaction 9:  NO + O + M => NO2 + M
    const amrex::Real k_f = 4710000000000 * exp((-2.87) * logT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[2] * sc[4]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[2] + g_RT[4] - g_RT[6])) * (refC) * (0.0);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[4] -= qdot;
    wdot[6] += qdot;
  }

  {
    // reaction 10:  NO2 + M => NO + O + M
    const amrex::Real k_f = 10800000000 * exp(-(33011.0132767896) * invT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[6]);
    const amrex::Real qr =
      Corr * k_f * exp(-(-g_RT[2] - g_RT[4] + g_RT[6])) * (refCinv) * (0.0);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[4] += qdot;
    wdot[6] -= qdot;
  }

  {
    // reaction 16:  N2O3 + M => NO + NO2 + M
    const amrex::Real k_f = 65000000 * exp(-(2626.79099550064) * invT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[7]);
    const amrex::Real qr =
      Corr * k_f * exp(-(-g_RT[4] - g_RT[6] + g_RT[7])) * (refCinv) * (0.0);
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[6] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 17:  NO + NO2 + M => N2O3 + M
    const amrex::Real k_f = 1.33e+21 * exp((-7.7) * logT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[4] * sc[6]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[4] + g_RT[6] - g_RT[7])) * (refC) * (0.0);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[6] -= qdot;
    wdot[7] += qdot;
  }

  {
    // reaction 0:  N2 + O <=> N + NO
    const amrex::Real k_f = 181000000 * exp(-(38395.4316009001) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[2]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] - g_RT[1] + g_RT[2] - g_RT[4])) * (sc[1] * sc[4]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[4] += qdot;
  }

  {
    // reaction 1:  NO + O <=> N + O2
    const amrex::Real k_f =
      687 * exp((1.13) * logT - (19222.8766337403) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[2] - g_RT[3] + g_RT[4])) * (sc[1] * sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[4] -= qdot;
  }

  {
    // reaction 6:  N2O + O => 2 NO
    const amrex::Real k_f = 90300000 * exp(-(13939.1016427907) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[5]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[4] += 2.000000 * qdot;
    wdot[5] -= qdot;
  }

  {
    // reaction 7:  2 NO => N2O + O
    const amrex::Real k_f = 3610000 * exp(-(32910.3699436287) * invT);
    const amrex::Real qf = k_f * ((sc[4] * sc[4]));
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[4] -= 2.000000 * qdot;
    wdot[5] += qdot;
  }

  {
    // reaction 8:  N2O + O <=> N2 + O2
    const amrex::Real k_f = 3670000 * exp(-(986304.664977251) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[5]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] + g_RT[2] - g_RT[3] + g_RT[5])) * (sc[0] * sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[5] -= qdot;
  }

  {
    // reaction 11:  NO2 + O => NO + O2
    const amrex::Real k_f = 3910000 * exp(-(-119.765566461523) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[6]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[4] += qdot;
    wdot[6] -= qdot;
  }

  {
    // reaction 12:  2 NO <=> N2 + O2
    const amrex::Real k_f = 12600000 * exp(-(30293.6432814441) * invT);
    const amrex::Real qf = k_f * ((sc[4] * sc[4]));
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] - g_RT[3] + 2.000000 * g_RT[4])) * (sc[0] * sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[3] += qdot;
    wdot[4] -= 2.000000 * qdot;
  }

  {
    // reaction 13:  N2O + NO => N2 + NO2
    const amrex::Real k_f = 18500000 * exp(-(24909.2249573336) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[5]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[4] -= qdot;
    wdot[5] -= qdot;
    wdot[6] += qdot;
  }

  {
    // reaction 14:  2 NO + O2 => 2 NO2
    const amrex::Real k_f = 0.0012 * exp(-(-528.377499094956) * invT);
    const amrex::Real qf = k_f * (sc[3] * (sc[4] * sc[4]));
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[4] -= 2.000000 * qdot;
    wdot[6] += 2.000000 * qdot;
  }

  {
    // reaction 15:  2 NO2 => 2 NO + O2
    const amrex::Real k_f = 1630000 * exp(-(13385.5633104055) * invT);
    const amrex::Real qf = k_f * ((sc[6] * sc[6]));
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[4] += 2.000000 * qdot;
    wdot[6] -= 2.000000 * qdot;
  }

  {
    // reaction 18:  2 NO2 => NO + NO3
    const amrex::Real k_f =
      9640 * exp((0.73) * logT - (10517.2283153186) * invT);
    const amrex::Real qf = k_f * ((sc[6] * sc[6]));
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[6] -= 2.000000 * qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 19:  NO + NO3 => 2 NO2
    const amrex::Real k_f = 10800000 * exp(-(-108.191583148015) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[8]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[6] += 2.000000 * qdot;
    wdot[8] -= qdot;
  }
}

// compute the production rate for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWC(const amrex::Real T, amrex::Real C[], amrex::Real wdot[])
{

  // convert to SI
  for (int id = 0; id < 9; ++id) {
    C[id] *= 1.0e6;
  }

  // convert to chemkin units
  productionRate(wdot, C, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
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
  amrex::Real c[9]; // temporary storage
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }
  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // multiply by 1e6 so c goes to SI
  PWORT *= 1e6;
  // Now compute conversion (and go to SI)
  for (int i = 0; i < 9; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
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
  amrex::Real c[9]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 9; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
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
  amrex::Real c[9]; // temporary storage

  // See Eq 8 with an extra 1e6 so c goes to SI
  for (int i = 0; i < 9; i++) {
    c[i] = 1e6 * rho * y[i] * imw(i);
  }

  // call productionRate
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
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
  amrex::Real c[9];   // temporary storage
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 28.014000; // N2
  XW += x[1] * 14.007000; // N
  XW += x[2] * 15.999000; // O
  XW += x[3] * 31.998000; // O2
  XW += x[4] * 30.006000; // NO
  XW += x[5] * 44.013000; // N2O
  XW += x[6] * 46.005000; // NO2
  XW += x[7] * 76.011000; // N2O3
  XW += x[8] * 62.004000; // NO3
  // Extra 1e6 factor to take c to SI
  ROW = 1e6 * rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 9; ++id) {
    c[id] = x[id] * ROW;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRG(int kcharge[])
{
  kcharge[0] = 0; // N2
  kcharge[1] = 0; // N
  kcharge[2] = 0; // O
  kcharge[3] = 0; // O2
  kcharge[4] = 0; // NO
  kcharge[5] = 0; // N2O
  kcharge[6] = 0; // NO2
  kcharge[7] = 0; // N2O3
  kcharge[8] = 0; // NO3
}

//  species charge per unit mass
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRGMASS(amrex::Real zk[])
{

  int kchrg[9];
  CKCHRG(kchrg);

  for (int id = 0; id < 9; ++id) {
    zk[id] = 6.02214076e+23 * 1.60217663e-19 * kchrg[id] * imw(id);
  }
}

// compute an approx to the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian_precond(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int HP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 100; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[9];
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
  for (int k = 0; k < 9; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[9];
  gibbs(g_RT, T);

  // compute the species enthalpy
  amrex::Real h_RT[9];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[9];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 2: O2 + M <=> 2 O + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[3];
  k_f = 1170000000 * exp(-(60385.9998965664) * invT);
  dlnkfdT = (60385.9998965664) * invT2;
  // reverse
  phi_r = (sc[2] * sc[2]);
  Kc = refC * exp(-2.000000 * g_RT[2] + g_RT[3]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3]) + (2.000000 * h_RT[2]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += 2 * q; // O
  wdot[3] -= q;     // O2
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor;
  dqdc[2] = q_nocor - k_r * 2.000000 * sc[2];
  dqdc[3] = q_nocor + k_f;
  dqdc[4] = q_nocor;
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  for (int k = 0; k < 9; k++) {
    J[10 * k + 2] += 2 * dqdc[k];
    J[10 * k + 3] -= dqdc[k];
  }
  J[92] += 2 * dqdT; // dwdot[O]/dT
  J[93] -= dqdT;     // dwdot[O2]/dT

  // reaction 3: N2 + M <=> 2 N + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[0];
  k_f = 6380000000 * exp(-(113726.966471867) * invT);
  dlnkfdT = (113726.966471867) * invT2;
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
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor;
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  for (int k = 0; k < 9; k++) {
    J[10 * k + 0] -= dqdc[k];
    J[10 * k + 1] += 2 * dqdc[k];
  }
  J[90] -= dqdT;     // dwdot[N2]/dT
  J[91] += 2 * dqdT; // dwdot[N]/dT

  // reaction 4: NO + M <=> N + O + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[4];
  k_f = 1450000000 * exp(-(74476.0665390985) * invT);
  dlnkfdT = (74476.0665390985) * invT2;
  // reverse
  phi_r = sc[1] * sc[2];
  Kc = refC * exp(-g_RT[1] - g_RT[2] + g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4]) + (h_RT[1] + h_RT[2]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // N
  wdot[2] += q; // O
  wdot[4] -= q; // NO
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor - k_r * sc[2];
  dqdc[2] = q_nocor - k_r * sc[1];
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor + k_f;
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  for (int k = 0; k < 9; k++) {
    J[10 * k + 1] += dqdc[k];
    J[10 * k + 2] += dqdc[k];
    J[10 * k + 4] -= dqdc[k];
  }
  J[91] += dqdT; // dwdot[N]/dT
  J[92] += dqdT; // dwdot[O]/dT
  J[94] -= dqdT; // dwdot[NO]/dT

  // reaction 5: N2O + M <=> N2 + O + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[5];
  k_f = 723000000000 * exp(-0.73 * logT - (31602.0066125364) * invT);
  dlnkfdT = -0.73 * invT + (31602.0066125364) * invT2;
  // reverse
  phi_r = sc[0] * sc[2];
  Kc = refC * exp(-g_RT[0] - g_RT[2] + g_RT[5]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[5]) + (h_RT[0] + h_RT[2]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // N2
  wdot[2] += q; // O
  wdot[5] -= q; // N2O
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor - k_r * sc[2];
  dqdc[1] = q_nocor;
  dqdc[2] = q_nocor - k_r * sc[0];
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor;
  dqdc[5] = q_nocor + k_f;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  for (int k = 0; k < 9; k++) {
    J[10 * k + 0] += dqdc[k];
    J[10 * k + 2] += dqdc[k];
    J[10 * k + 5] -= dqdc[k];
  }
  J[90] += dqdT; // dwdot[N2]/dT
  J[92] += dqdT; // dwdot[O]/dT
  J[95] -= dqdT; // dwdot[N2O]/dT

  // reaction 9: NO + O + M => NO2 + M
  // a third-body and non-pressure-fall-off reaction
  // Irreversible reaction in _ajac_reaction may not work
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[2] * sc[4];
  k_f = 4710000000000 * exp(-2.87 * logT);
  dlnkfdT = -2.87 * invT;
  // rate of progress
  q_nocor = k_f * phi_f;
  q = alpha * q_nocor;
  dqdT = alpha * dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] -= q; // O
  wdot[4] -= q; // NO
  wdot[6] += q; // NO2
  // for convenience
  k_f *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor;
  dqdc[2] = q_nocor + k_f * sc[4];
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor + k_f * sc[2];
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  for (int k = 0; k < 9; k++) {
    J[10 * k + 2] -= dqdc[k];
    J[10 * k + 4] -= dqdc[k];
    J[10 * k + 6] += dqdc[k];
  }
  J[92] -= dqdT; // dwdot[O]/dT
  J[94] -= dqdT; // dwdot[NO]/dT
  J[96] += dqdT; // dwdot[NO2]/dT

  // reaction 10: NO2 + M => NO + O + M
  // a third-body and non-pressure-fall-off reaction
  // Irreversible reaction in _ajac_reaction may not work
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[6];
  k_f = 10800000000 * exp(-(33011.0132767896) * invT);
  dlnkfdT = (33011.0132767896) * invT2;
  // rate of progress
  q_nocor = k_f * phi_f;
  q = alpha * q_nocor;
  dqdT = alpha * dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += q; // O
  wdot[4] += q; // NO
  wdot[6] -= q; // NO2
  // for convenience
  k_f *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor;
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor;
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor + k_f;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  for (int k = 0; k < 9; k++) {
    J[10 * k + 2] += dqdc[k];
    J[10 * k + 4] += dqdc[k];
    J[10 * k + 6] -= dqdc[k];
  }
  J[92] += dqdT; // dwdot[O]/dT
  J[94] += dqdT; // dwdot[NO]/dT
  J[96] -= dqdT; // dwdot[NO2]/dT

  // reaction 16: N2O3 + M => NO + NO2 + M
  // a third-body and non-pressure-fall-off reaction
  // Irreversible reaction in _ajac_reaction may not work
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[7];
  k_f = 65000000 * exp(-(2626.79099550064) * invT);
  dlnkfdT = (2626.79099550064) * invT2;
  // rate of progress
  q_nocor = k_f * phi_f;
  q = alpha * q_nocor;
  dqdT = alpha * dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[4] += q; // NO
  wdot[6] += q; // NO2
  wdot[7] -= q; // N2O3
  // for convenience
  k_f *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor;
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor;
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor + k_f;
  dqdc[8] = q_nocor;
  for (int k = 0; k < 9; k++) {
    J[10 * k + 4] += dqdc[k];
    J[10 * k + 6] += dqdc[k];
    J[10 * k + 7] -= dqdc[k];
  }
  J[94] += dqdT; // dwdot[NO]/dT
  J[96] += dqdT; // dwdot[NO2]/dT
  J[97] -= dqdT; // dwdot[N2O3]/dT

  // reaction 17: NO + NO2 + M => N2O3 + M
  // a third-body and non-pressure-fall-off reaction
  // Irreversible reaction in _ajac_reaction may not work
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[4] * sc[6];
  k_f = 1.33e+21 * exp(-7.7 * logT);
  dlnkfdT = -7.7 * invT;
  // rate of progress
  q_nocor = k_f * phi_f;
  q = alpha * q_nocor;
  dqdT = alpha * dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[4] -= q; // NO
  wdot[6] -= q; // NO2
  wdot[7] += q; // N2O3
  // for convenience
  k_f *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor;
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor + k_f * sc[6];
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor + k_f * sc[4];
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  for (int k = 0; k < 9; k++) {
    J[10 * k + 4] -= dqdc[k];
    J[10 * k + 6] -= dqdc[k];
    J[10 * k + 7] += dqdc[k];
  }
  J[94] -= dqdT; // dwdot[NO]/dT
  J[96] -= dqdT; // dwdot[NO2]/dT
  J[97] += dqdT; // dwdot[N2O3]/dT

  // reaction 0: N2 + O <=> N + NO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[2];
  k_f = 181000000 * exp(-(38395.4316009001) * invT);
  dlnkfdT = (38395.4316009001) * invT2;
  // reverse
  phi_r = sc[1] * sc[4];
  Kc = exp(g_RT[0] - g_RT[1] + g_RT[2] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[2]) + (h_RT[1] + h_RT[4]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // N2
  wdot[1] += q; // N
  wdot[2] -= q; // O
  wdot[4] += q; // NO
  // d()/d[N2]
  dqdci = +k_f * sc[2];
  J[0] -= dqdci; // dwdot[N2]/d[N2]
  J[1] += dqdci; // dwdot[N]/d[N2]
  J[2] -= dqdci; // dwdot[O]/d[N2]
  J[4] += dqdci; // dwdot[NO]/d[N2]
  // d()/d[N]
  dqdci = -k_r * sc[4];
  J[10] -= dqdci; // dwdot[N2]/d[N]
  J[11] += dqdci; // dwdot[N]/d[N]
  J[12] -= dqdci; // dwdot[O]/d[N]
  J[14] += dqdci; // dwdot[NO]/d[N]
  // d()/d[O]
  dqdci = +k_f * sc[0];
  J[20] -= dqdci; // dwdot[N2]/d[O]
  J[21] += dqdci; // dwdot[N]/d[O]
  J[22] -= dqdci; // dwdot[O]/d[O]
  J[24] += dqdci; // dwdot[NO]/d[O]
  // d()/d[NO]
  dqdci = -k_r * sc[1];
  J[40] -= dqdci; // dwdot[N2]/d[NO]
  J[41] += dqdci; // dwdot[N]/d[NO]
  J[42] -= dqdci; // dwdot[O]/d[NO]
  J[44] += dqdci; // dwdot[NO]/d[NO]
  // d()/dT
  J[90] -= dqdT; // dwdot[N2]/dT
  J[91] += dqdT; // dwdot[N]/dT
  J[92] -= dqdT; // dwdot[O]/dT
  J[94] += dqdT; // dwdot[NO]/dT

  // reaction 1: NO + O <=> N + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[4];
  k_f = 687 * exp(1.13 * logT - (19222.8766337403) * invT);
  dlnkfdT = 1.13 * invT + (19222.8766337403) * invT2;
  // reverse
  phi_r = sc[1] * sc[3];
  Kc = exp(-g_RT[1] + g_RT[2] - g_RT[3] + g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[4]) + (h_RT[1] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // N
  wdot[2] -= q; // O
  wdot[3] += q; // O2
  wdot[4] -= q; // NO
  // d()/d[N]
  dqdci = -k_r * sc[3];
  J[11] += dqdci; // dwdot[N]/d[N]
  J[12] -= dqdci; // dwdot[O]/d[N]
  J[13] += dqdci; // dwdot[O2]/d[N]
  J[14] -= dqdci; // dwdot[NO]/d[N]
  // d()/d[O]
  dqdci = +k_f * sc[4];
  J[21] += dqdci; // dwdot[N]/d[O]
  J[22] -= dqdci; // dwdot[O]/d[O]
  J[23] += dqdci; // dwdot[O2]/d[O]
  J[24] -= dqdci; // dwdot[NO]/d[O]
  // d()/d[O2]
  dqdci = -k_r * sc[1];
  J[31] += dqdci; // dwdot[N]/d[O2]
  J[32] -= dqdci; // dwdot[O]/d[O2]
  J[33] += dqdci; // dwdot[O2]/d[O2]
  J[34] -= dqdci; // dwdot[NO]/d[O2]
  // d()/d[NO]
  dqdci = +k_f * sc[2];
  J[41] += dqdci; // dwdot[N]/d[NO]
  J[42] -= dqdci; // dwdot[O]/d[NO]
  J[43] += dqdci; // dwdot[O2]/d[NO]
  J[44] -= dqdci; // dwdot[NO]/d[NO]
  // d()/dT
  J[91] += dqdT; // dwdot[N]/dT
  J[92] -= dqdT; // dwdot[O]/dT
  J[93] += dqdT; // dwdot[O2]/dT
  J[94] -= dqdT; // dwdot[NO]/dT

  // reaction 6: N2O + O => 2 NO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[5];
  k_f = 90300000 * exp(-(13939.1016427907) * invT);
  dlnkfdT = (13939.1016427907) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] -= q;     // O
  wdot[4] += 2 * q; // NO
  wdot[5] -= q;     // N2O
  // d()/d[O]
  dqdci = +k_f * sc[5];
  J[22] -= dqdci;     // dwdot[O]/d[O]
  J[24] += 2 * dqdci; // dwdot[NO]/d[O]
  J[25] -= dqdci;     // dwdot[N2O]/d[O]
  // d()/d[N2O]
  dqdci = +k_f * sc[2];
  J[52] -= dqdci;     // dwdot[O]/d[N2O]
  J[54] += 2 * dqdci; // dwdot[NO]/d[N2O]
  J[55] -= dqdci;     // dwdot[N2O]/d[N2O]
  // d()/dT
  J[92] -= dqdT;     // dwdot[O]/dT
  J[94] += 2 * dqdT; // dwdot[NO]/dT
  J[95] -= dqdT;     // dwdot[N2O]/dT

  // reaction 7: 2 NO => N2O + O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[4] * sc[4]);
  k_f = 3610000 * exp(-(32910.3699436287) * invT);
  dlnkfdT = (32910.3699436287) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += q;     // O
  wdot[4] -= 2 * q; // NO
  wdot[5] += q;     // N2O
  // d()/d[NO]
  dqdci = +k_f * 2.000000 * sc[4];
  J[42] += dqdci;      // dwdot[O]/d[NO]
  J[44] += -2 * dqdci; // dwdot[NO]/d[NO]
  J[45] += dqdci;      // dwdot[N2O]/d[NO]
  // d()/dT
  J[92] += dqdT;      // dwdot[O]/dT
  J[94] += -2 * dqdT; // dwdot[NO]/dT
  J[95] += dqdT;      // dwdot[N2O]/dT

  // reaction 8: N2O + O <=> N2 + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[5];
  k_f = 3670000 * exp(-(986304.664977251) * invT);
  dlnkfdT = (986304.664977251) * invT2;
  // reverse
  phi_r = sc[0] * sc[3];
  Kc = exp(-g_RT[0] + g_RT[2] - g_RT[3] + g_RT[5]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[5]) + (h_RT[0] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // N2
  wdot[2] -= q; // O
  wdot[3] += q; // O2
  wdot[5] -= q; // N2O
  // d()/d[N2]
  dqdci = -k_r * sc[3];
  J[0] += dqdci; // dwdot[N2]/d[N2]
  J[2] -= dqdci; // dwdot[O]/d[N2]
  J[3] += dqdci; // dwdot[O2]/d[N2]
  J[5] -= dqdci; // dwdot[N2O]/d[N2]
  // d()/d[O]
  dqdci = +k_f * sc[5];
  J[20] += dqdci; // dwdot[N2]/d[O]
  J[22] -= dqdci; // dwdot[O]/d[O]
  J[23] += dqdci; // dwdot[O2]/d[O]
  J[25] -= dqdci; // dwdot[N2O]/d[O]
  // d()/d[O2]
  dqdci = -k_r * sc[0];
  J[30] += dqdci; // dwdot[N2]/d[O2]
  J[32] -= dqdci; // dwdot[O]/d[O2]
  J[33] += dqdci; // dwdot[O2]/d[O2]
  J[35] -= dqdci; // dwdot[N2O]/d[O2]
  // d()/d[N2O]
  dqdci = +k_f * sc[2];
  J[50] += dqdci; // dwdot[N2]/d[N2O]
  J[52] -= dqdci; // dwdot[O]/d[N2O]
  J[53] += dqdci; // dwdot[O2]/d[N2O]
  J[55] -= dqdci; // dwdot[N2O]/d[N2O]
  // d()/dT
  J[90] += dqdT; // dwdot[N2]/dT
  J[92] -= dqdT; // dwdot[O]/dT
  J[93] += dqdT; // dwdot[O2]/dT
  J[95] -= dqdT; // dwdot[N2O]/dT

  // reaction 11: NO2 + O => NO + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[6];
  k_f = 3910000 * exp(-(-119.765566461523) * invT);
  dlnkfdT = (-119.765566461523) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] -= q; // O
  wdot[3] += q; // O2
  wdot[4] += q; // NO
  wdot[6] -= q; // NO2
  // d()/d[O]
  dqdci = +k_f * sc[6];
  J[22] -= dqdci; // dwdot[O]/d[O]
  J[23] += dqdci; // dwdot[O2]/d[O]
  J[24] += dqdci; // dwdot[NO]/d[O]
  J[26] -= dqdci; // dwdot[NO2]/d[O]
  // d()/d[NO2]
  dqdci = +k_f * sc[2];
  J[62] -= dqdci; // dwdot[O]/d[NO2]
  J[63] += dqdci; // dwdot[O2]/d[NO2]
  J[64] += dqdci; // dwdot[NO]/d[NO2]
  J[66] -= dqdci; // dwdot[NO2]/d[NO2]
  // d()/dT
  J[92] -= dqdT; // dwdot[O]/dT
  J[93] += dqdT; // dwdot[O2]/dT
  J[94] += dqdT; // dwdot[NO]/dT
  J[96] -= dqdT; // dwdot[NO2]/dT

  // reaction 12: 2 NO <=> N2 + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[4] * sc[4]);
  k_f = 12600000 * exp(-(30293.6432814441) * invT);
  dlnkfdT = (30293.6432814441) * invT2;
  // reverse
  phi_r = sc[0] * sc[3];
  Kc = exp(-g_RT[0] - g_RT[3] + 2.000000 * g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[4]) + (h_RT[0] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q;     // N2
  wdot[3] += q;     // O2
  wdot[4] -= 2 * q; // NO
  // d()/d[N2]
  dqdci = -k_r * sc[3];
  J[0] += dqdci;      // dwdot[N2]/d[N2]
  J[3] += dqdci;      // dwdot[O2]/d[N2]
  J[4] += -2 * dqdci; // dwdot[NO]/d[N2]
  // d()/d[O2]
  dqdci = -k_r * sc[0];
  J[30] += dqdci;      // dwdot[N2]/d[O2]
  J[33] += dqdci;      // dwdot[O2]/d[O2]
  J[34] += -2 * dqdci; // dwdot[NO]/d[O2]
  // d()/d[NO]
  dqdci = +k_f * 2.000000 * sc[4];
  J[40] += dqdci;      // dwdot[N2]/d[NO]
  J[43] += dqdci;      // dwdot[O2]/d[NO]
  J[44] += -2 * dqdci; // dwdot[NO]/d[NO]
  // d()/dT
  J[90] += dqdT;      // dwdot[N2]/dT
  J[93] += dqdT;      // dwdot[O2]/dT
  J[94] += -2 * dqdT; // dwdot[NO]/dT

  // reaction 13: N2O + NO => N2 + NO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[5];
  k_f = 18500000 * exp(-(24909.2249573336) * invT);
  dlnkfdT = (24909.2249573336) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[0] += q; // N2
  wdot[4] -= q; // NO
  wdot[5] -= q; // N2O
  wdot[6] += q; // NO2
  // d()/d[NO]
  dqdci = +k_f * sc[5];
  J[40] += dqdci; // dwdot[N2]/d[NO]
  J[44] -= dqdci; // dwdot[NO]/d[NO]
  J[45] -= dqdci; // dwdot[N2O]/d[NO]
  J[46] += dqdci; // dwdot[NO2]/d[NO]
  // d()/d[N2O]
  dqdci = +k_f * sc[4];
  J[50] += dqdci; // dwdot[N2]/d[N2O]
  J[54] -= dqdci; // dwdot[NO]/d[N2O]
  J[55] -= dqdci; // dwdot[N2O]/d[N2O]
  J[56] += dqdci; // dwdot[NO2]/d[N2O]
  // d()/dT
  J[90] += dqdT; // dwdot[N2]/dT
  J[94] -= dqdT; // dwdot[NO]/dT
  J[95] -= dqdT; // dwdot[N2O]/dT
  J[96] += dqdT; // dwdot[NO2]/dT

  // reaction 14: 2 NO + O2 => 2 NO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * (sc[4] * sc[4]);
  k_f = 0.0012 * exp(-(-528.377499094956) * invT);
  dlnkfdT = (-528.377499094956) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[3] -= q;     // O2
  wdot[4] -= 2 * q; // NO
  wdot[6] += 2 * q; // NO2
  // d()/d[O2]
  dqdci = +k_f * sc[4] * sc[4];
  J[33] -= dqdci;      // dwdot[O2]/d[O2]
  J[34] += -2 * dqdci; // dwdot[NO]/d[O2]
  J[36] += 2 * dqdci;  // dwdot[NO2]/d[O2]
  // d()/d[NO]
  dqdci = +k_f * sc[3] * 2.000000 * sc[4];
  J[43] -= dqdci;      // dwdot[O2]/d[NO]
  J[44] += -2 * dqdci; // dwdot[NO]/d[NO]
  J[46] += 2 * dqdci;  // dwdot[NO2]/d[NO]
  // d()/dT
  J[93] -= dqdT;      // dwdot[O2]/dT
  J[94] += -2 * dqdT; // dwdot[NO]/dT
  J[96] += 2 * dqdT;  // dwdot[NO2]/dT

  // reaction 15: 2 NO2 => 2 NO + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[6] * sc[6]);
  k_f = 1630000 * exp(-(13385.5633104055) * invT);
  dlnkfdT = (13385.5633104055) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[3] += q;     // O2
  wdot[4] += 2 * q; // NO
  wdot[6] -= 2 * q; // NO2
  // d()/d[NO2]
  dqdci = +k_f * 2.000000 * sc[6];
  J[63] += dqdci;      // dwdot[O2]/d[NO2]
  J[64] += 2 * dqdci;  // dwdot[NO]/d[NO2]
  J[66] += -2 * dqdci; // dwdot[NO2]/d[NO2]
  // d()/dT
  J[93] += dqdT;      // dwdot[O2]/dT
  J[94] += 2 * dqdT;  // dwdot[NO]/dT
  J[96] += -2 * dqdT; // dwdot[NO2]/dT

  // reaction 18: 2 NO2 => NO + NO3
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[6] * sc[6]);
  k_f = 9640 * exp(0.73 * logT - (10517.2283153186) * invT);
  dlnkfdT = 0.73 * invT + (10517.2283153186) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[4] += q;     // NO
  wdot[6] -= 2 * q; // NO2
  wdot[8] += q;     // NO3
  // d()/d[NO2]
  dqdci = +k_f * 2.000000 * sc[6];
  J[64] += dqdci;      // dwdot[NO]/d[NO2]
  J[66] += -2 * dqdci; // dwdot[NO2]/d[NO2]
  J[68] += dqdci;      // dwdot[NO3]/d[NO2]
  // d()/dT
  J[94] += dqdT;      // dwdot[NO]/dT
  J[96] += -2 * dqdT; // dwdot[NO2]/dT
  J[98] += dqdT;      // dwdot[NO3]/dT

  // reaction 19: NO + NO3 => 2 NO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[8];
  k_f = 10800000 * exp(-(-108.191583148015) * invT);
  dlnkfdT = (-108.191583148015) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[4] -= q;     // NO
  wdot[6] += 2 * q; // NO2
  wdot[8] -= q;     // NO3
  // d()/d[NO]
  dqdci = +k_f * sc[8];
  J[44] -= dqdci;     // dwdot[NO]/d[NO]
  J[46] += 2 * dqdci; // dwdot[NO2]/d[NO]
  J[48] -= dqdci;     // dwdot[NO3]/d[NO]
  // d()/d[NO3]
  dqdci = +k_f * sc[4];
  J[84] -= dqdci;     // dwdot[NO]/d[NO3]
  J[86] += 2 * dqdci; // dwdot[NO2]/d[NO3]
  J[88] -= dqdci;     // dwdot[NO3]/d[NO3]
  // d()/dT
  J[94] -= dqdT;     // dwdot[NO]/dT
  J[96] += 2 * dqdT; // dwdot[NO2]/dT
  J[98] -= dqdT;     // dwdot[NO3]/dT

  amrex::Real c_R[9], dcRdT[9], e_RT[9];
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
  for (int k = 0; k < 9; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[90 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 9; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 9; ++m) {
      dehmixdc += eh_RT[m] * J[k * 10 + m];
    }
    J[k * 10 + 9] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[99] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
#else
  amrex::Abort();
#endif
}

// compute an approx to the reaction Jacobian (for preconditioning)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT_SIMPLIFIED(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real* Tp, const int* HP)
{
  amrex::Real c[9];

  for (int k = 0; k < 9; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian_precond(J, c, *Tp, *HP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 9; k++) {
    J[90 + k] *= 1.e-6;
    J[k * 10 + 9] *= 1.e6;
  }
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int consP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 100; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[9];
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
  for (int k = 0; k < 9; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[9];
  gibbs(g_RT, T);

  // compute the species enthalpy
  amrex::Real h_RT[9];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[9];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 2: O2 + M <=> 2 O + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[3];
  k_f = 1170000000 * exp(-(60385.9998965664) * invT);
  dlnkfdT = (60385.9998965664) * invT2;
  // reverse
  phi_r = (sc[2] * sc[2]);
  Kc = refC * exp(-2.000000 * g_RT[2] + g_RT[3]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3]) + (2.000000 * h_RT[2]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += 2 * q; // O
  wdot[3] -= q;     // O2
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[N2]
    dqdci = (1 - 1) * q_nocor;
    J[2] += 2 * dqdci; // dwdot[O]/d[N2]
    J[3] -= dqdci;     // dwdot[O2]/d[N2]
    // d()/d[N]
    dqdci = (1 - 1) * q_nocor;
    J[12] += 2 * dqdci; // dwdot[O]/d[N]
    J[13] -= dqdci;     // dwdot[O2]/d[N]
    // d()/d[O]
    dqdci = (1 - 1) * q_nocor - k_r * 2.000000 * sc[2];
    J[22] += 2 * dqdci; // dwdot[O]/d[O]
    J[23] -= dqdci;     // dwdot[O2]/d[O]
    // d()/d[O2]
    dqdci = (1 - 1) * q_nocor + k_f;
    J[32] += 2 * dqdci; // dwdot[O]/d[O2]
    J[33] -= dqdci;     // dwdot[O2]/d[O2]
    // d()/d[NO]
    dqdci = (1 - 1) * q_nocor;
    J[42] += 2 * dqdci; // dwdot[O]/d[NO]
    J[43] -= dqdci;     // dwdot[O2]/d[NO]
    // d()/d[N2O]
    dqdci = (1 - 1) * q_nocor;
    J[52] += 2 * dqdci; // dwdot[O]/d[N2O]
    J[53] -= dqdci;     // dwdot[O2]/d[N2O]
    // d()/d[NO2]
    dqdci = (1 - 1) * q_nocor;
    J[62] += 2 * dqdci; // dwdot[O]/d[NO2]
    J[63] -= dqdci;     // dwdot[O2]/d[NO2]
    // d()/d[N2O3]
    dqdci = (1 - 1) * q_nocor;
    J[72] += 2 * dqdci; // dwdot[O]/d[N2O3]
    J[73] -= dqdci;     // dwdot[O2]/d[N2O3]
    // d()/d[NO3]
    dqdci = (1 - 1) * q_nocor;
    J[82] += 2 * dqdci; // dwdot[O]/d[NO3]
    J[83] -= dqdci;     // dwdot[O2]/d[NO3]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor;
    dqdc[2] = q_nocor - k_r * 2.000000 * sc[2];
    dqdc[3] = q_nocor + k_f;
    dqdc[4] = q_nocor;
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    for (int k = 0; k < 9; k++) {
      J[10 * k + 2] += 2 * dqdc[k];
      J[10 * k + 3] -= dqdc[k];
    }
  }
  J[92] += 2 * dqdT; // dwdot[O]/dT
  J[93] -= dqdT;     // dwdot[O2]/dT

  // reaction 3: N2 + M <=> 2 N + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[0];
  k_f = 6380000000 * exp(-(113726.966471867) * invT);
  dlnkfdT = (113726.966471867) * invT2;
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
    J[10] -= dqdci;     // dwdot[N2]/d[N]
    J[11] += 2 * dqdci; // dwdot[N]/d[N]
    // d()/d[O]
    dqdci = (1 - 1) * q_nocor;
    J[20] -= dqdci;     // dwdot[N2]/d[O]
    J[21] += 2 * dqdci; // dwdot[N]/d[O]
    // d()/d[O2]
    dqdci = (1 - 1) * q_nocor;
    J[30] -= dqdci;     // dwdot[N2]/d[O2]
    J[31] += 2 * dqdci; // dwdot[N]/d[O2]
    // d()/d[NO]
    dqdci = (1 - 1) * q_nocor;
    J[40] -= dqdci;     // dwdot[N2]/d[NO]
    J[41] += 2 * dqdci; // dwdot[N]/d[NO]
    // d()/d[N2O]
    dqdci = (1 - 1) * q_nocor;
    J[50] -= dqdci;     // dwdot[N2]/d[N2O]
    J[51] += 2 * dqdci; // dwdot[N]/d[N2O]
    // d()/d[NO2]
    dqdci = (1 - 1) * q_nocor;
    J[60] -= dqdci;     // dwdot[N2]/d[NO2]
    J[61] += 2 * dqdci; // dwdot[N]/d[NO2]
    // d()/d[N2O3]
    dqdci = (1 - 1) * q_nocor;
    J[70] -= dqdci;     // dwdot[N2]/d[N2O3]
    J[71] += 2 * dqdci; // dwdot[N]/d[N2O3]
    // d()/d[NO3]
    dqdci = (1 - 1) * q_nocor;
    J[80] -= dqdci;     // dwdot[N2]/d[NO3]
    J[81] += 2 * dqdci; // dwdot[N]/d[NO3]
  } else {
    dqdc[0] = q_nocor + k_f;
    dqdc[1] = q_nocor - k_r * 2.000000 * sc[1];
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor;
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    for (int k = 0; k < 9; k++) {
      J[10 * k + 0] -= dqdc[k];
      J[10 * k + 1] += 2 * dqdc[k];
    }
  }
  J[90] -= dqdT;     // dwdot[N2]/dT
  J[91] += 2 * dqdT; // dwdot[N]/dT

  // reaction 4: NO + M <=> N + O + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[4];
  k_f = 1450000000 * exp(-(74476.0665390985) * invT);
  dlnkfdT = (74476.0665390985) * invT2;
  // reverse
  phi_r = sc[1] * sc[2];
  Kc = refC * exp(-g_RT[1] - g_RT[2] + g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4]) + (h_RT[1] + h_RT[2]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // N
  wdot[2] += q; // O
  wdot[4] -= q; // NO
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[N2]
    dqdci = (1 - 1) * q_nocor;
    J[1] += dqdci; // dwdot[N]/d[N2]
    J[2] += dqdci; // dwdot[O]/d[N2]
    J[4] -= dqdci; // dwdot[NO]/d[N2]
    // d()/d[N]
    dqdci = (1 - 1) * q_nocor - k_r * sc[2];
    J[11] += dqdci; // dwdot[N]/d[N]
    J[12] += dqdci; // dwdot[O]/d[N]
    J[14] -= dqdci; // dwdot[NO]/d[N]
    // d()/d[O]
    dqdci = (1 - 1) * q_nocor - k_r * sc[1];
    J[21] += dqdci; // dwdot[N]/d[O]
    J[22] += dqdci; // dwdot[O]/d[O]
    J[24] -= dqdci; // dwdot[NO]/d[O]
    // d()/d[O2]
    dqdci = (1 - 1) * q_nocor;
    J[31] += dqdci; // dwdot[N]/d[O2]
    J[32] += dqdci; // dwdot[O]/d[O2]
    J[34] -= dqdci; // dwdot[NO]/d[O2]
    // d()/d[NO]
    dqdci = (1 - 1) * q_nocor + k_f;
    J[41] += dqdci; // dwdot[N]/d[NO]
    J[42] += dqdci; // dwdot[O]/d[NO]
    J[44] -= dqdci; // dwdot[NO]/d[NO]
    // d()/d[N2O]
    dqdci = (1 - 1) * q_nocor;
    J[51] += dqdci; // dwdot[N]/d[N2O]
    J[52] += dqdci; // dwdot[O]/d[N2O]
    J[54] -= dqdci; // dwdot[NO]/d[N2O]
    // d()/d[NO2]
    dqdci = (1 - 1) * q_nocor;
    J[61] += dqdci; // dwdot[N]/d[NO2]
    J[62] += dqdci; // dwdot[O]/d[NO2]
    J[64] -= dqdci; // dwdot[NO]/d[NO2]
    // d()/d[N2O3]
    dqdci = (1 - 1) * q_nocor;
    J[71] += dqdci; // dwdot[N]/d[N2O3]
    J[72] += dqdci; // dwdot[O]/d[N2O3]
    J[74] -= dqdci; // dwdot[NO]/d[N2O3]
    // d()/d[NO3]
    dqdci = (1 - 1) * q_nocor;
    J[81] += dqdci; // dwdot[N]/d[NO3]
    J[82] += dqdci; // dwdot[O]/d[NO3]
    J[84] -= dqdci; // dwdot[NO]/d[NO3]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor - k_r * sc[2];
    dqdc[2] = q_nocor - k_r * sc[1];
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor + k_f;
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    for (int k = 0; k < 9; k++) {
      J[10 * k + 1] += dqdc[k];
      J[10 * k + 2] += dqdc[k];
      J[10 * k + 4] -= dqdc[k];
    }
  }
  J[91] += dqdT; // dwdot[N]/dT
  J[92] += dqdT; // dwdot[O]/dT
  J[94] -= dqdT; // dwdot[NO]/dT

  // reaction 5: N2O + M <=> N2 + O + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[5];
  k_f = 723000000000 * exp(-0.73 * logT - (31602.0066125364) * invT);
  dlnkfdT = -0.73 * invT + (31602.0066125364) * invT2;
  // reverse
  phi_r = sc[0] * sc[2];
  Kc = refC * exp(-g_RT[0] - g_RT[2] + g_RT[5]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[5]) + (h_RT[0] + h_RT[2]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // N2
  wdot[2] += q; // O
  wdot[5] -= q; // N2O
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[N2]
    dqdci = (1 - 1) * q_nocor - k_r * sc[2];
    J[0] += dqdci; // dwdot[N2]/d[N2]
    J[2] += dqdci; // dwdot[O]/d[N2]
    J[5] -= dqdci; // dwdot[N2O]/d[N2]
    // d()/d[N]
    dqdci = (1 - 1) * q_nocor;
    J[10] += dqdci; // dwdot[N2]/d[N]
    J[12] += dqdci; // dwdot[O]/d[N]
    J[15] -= dqdci; // dwdot[N2O]/d[N]
    // d()/d[O]
    dqdci = (1 - 1) * q_nocor - k_r * sc[0];
    J[20] += dqdci; // dwdot[N2]/d[O]
    J[22] += dqdci; // dwdot[O]/d[O]
    J[25] -= dqdci; // dwdot[N2O]/d[O]
    // d()/d[O2]
    dqdci = (1 - 1) * q_nocor;
    J[30] += dqdci; // dwdot[N2]/d[O2]
    J[32] += dqdci; // dwdot[O]/d[O2]
    J[35] -= dqdci; // dwdot[N2O]/d[O2]
    // d()/d[NO]
    dqdci = (1 - 1) * q_nocor;
    J[40] += dqdci; // dwdot[N2]/d[NO]
    J[42] += dqdci; // dwdot[O]/d[NO]
    J[45] -= dqdci; // dwdot[N2O]/d[NO]
    // d()/d[N2O]
    dqdci = (1 - 1) * q_nocor + k_f;
    J[50] += dqdci; // dwdot[N2]/d[N2O]
    J[52] += dqdci; // dwdot[O]/d[N2O]
    J[55] -= dqdci; // dwdot[N2O]/d[N2O]
    // d()/d[NO2]
    dqdci = (1 - 1) * q_nocor;
    J[60] += dqdci; // dwdot[N2]/d[NO2]
    J[62] += dqdci; // dwdot[O]/d[NO2]
    J[65] -= dqdci; // dwdot[N2O]/d[NO2]
    // d()/d[N2O3]
    dqdci = (1 - 1) * q_nocor;
    J[70] += dqdci; // dwdot[N2]/d[N2O3]
    J[72] += dqdci; // dwdot[O]/d[N2O3]
    J[75] -= dqdci; // dwdot[N2O]/d[N2O3]
    // d()/d[NO3]
    dqdci = (1 - 1) * q_nocor;
    J[80] += dqdci; // dwdot[N2]/d[NO3]
    J[82] += dqdci; // dwdot[O]/d[NO3]
    J[85] -= dqdci; // dwdot[N2O]/d[NO3]
  } else {
    dqdc[0] = q_nocor - k_r * sc[2];
    dqdc[1] = q_nocor;
    dqdc[2] = q_nocor - k_r * sc[0];
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor;
    dqdc[5] = q_nocor + k_f;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    for (int k = 0; k < 9; k++) {
      J[10 * k + 0] += dqdc[k];
      J[10 * k + 2] += dqdc[k];
      J[10 * k + 5] -= dqdc[k];
    }
  }
  J[90] += dqdT; // dwdot[N2]/dT
  J[92] += dqdT; // dwdot[O]/dT
  J[95] -= dqdT; // dwdot[N2O]/dT

  // reaction 9: NO + O + M => NO2 + M
  // a third-body and non-pressure-fall-off reaction
  // Irreversible reaction in _ajac_reaction may not work
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[2] * sc[4];
  k_f = 4710000000000 * exp(-2.87 * logT);
  dlnkfdT = -2.87 * invT;
  // rate of progress
  q_nocor = k_f * phi_f;
  q = alpha * q_nocor;
  dqdT = alpha * dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] -= q; // O
  wdot[4] -= q; // NO
  wdot[6] += q; // NO2
  // for convenience
  k_f *= alpha;
  if (consP == 1) {
    // d()/d[N2]
    dqdci = (1 - 1) * q_nocor;
    J[2] -= dqdci; // dwdot[O]/d[N2]
    J[4] -= dqdci; // dwdot[NO]/d[N2]
    J[6] += dqdci; // dwdot[NO2]/d[N2]
    // d()/d[N]
    dqdci = (1 - 1) * q_nocor;
    J[12] -= dqdci; // dwdot[O]/d[N]
    J[14] -= dqdci; // dwdot[NO]/d[N]
    J[16] += dqdci; // dwdot[NO2]/d[N]
    // d()/d[O]
    dqdci = (1 - 1) * q_nocor + k_f * sc[4];
    J[22] -= dqdci; // dwdot[O]/d[O]
    J[24] -= dqdci; // dwdot[NO]/d[O]
    J[26] += dqdci; // dwdot[NO2]/d[O]
    // d()/d[O2]
    dqdci = (1 - 1) * q_nocor;
    J[32] -= dqdci; // dwdot[O]/d[O2]
    J[34] -= dqdci; // dwdot[NO]/d[O2]
    J[36] += dqdci; // dwdot[NO2]/d[O2]
    // d()/d[NO]
    dqdci = (1 - 1) * q_nocor + k_f * sc[2];
    J[42] -= dqdci; // dwdot[O]/d[NO]
    J[44] -= dqdci; // dwdot[NO]/d[NO]
    J[46] += dqdci; // dwdot[NO2]/d[NO]
    // d()/d[N2O]
    dqdci = (1 - 1) * q_nocor;
    J[52] -= dqdci; // dwdot[O]/d[N2O]
    J[54] -= dqdci; // dwdot[NO]/d[N2O]
    J[56] += dqdci; // dwdot[NO2]/d[N2O]
    // d()/d[NO2]
    dqdci = (1 - 1) * q_nocor;
    J[62] -= dqdci; // dwdot[O]/d[NO2]
    J[64] -= dqdci; // dwdot[NO]/d[NO2]
    J[66] += dqdci; // dwdot[NO2]/d[NO2]
    // d()/d[N2O3]
    dqdci = (1 - 1) * q_nocor;
    J[72] -= dqdci; // dwdot[O]/d[N2O3]
    J[74] -= dqdci; // dwdot[NO]/d[N2O3]
    J[76] += dqdci; // dwdot[NO2]/d[N2O3]
    // d()/d[NO3]
    dqdci = (1 - 1) * q_nocor;
    J[82] -= dqdci; // dwdot[O]/d[NO3]
    J[84] -= dqdci; // dwdot[NO]/d[NO3]
    J[86] += dqdci; // dwdot[NO2]/d[NO3]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor;
    dqdc[2] = q_nocor + k_f * sc[4];
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor + k_f * sc[2];
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    for (int k = 0; k < 9; k++) {
      J[10 * k + 2] -= dqdc[k];
      J[10 * k + 4] -= dqdc[k];
      J[10 * k + 6] += dqdc[k];
    }
  }
  J[92] -= dqdT; // dwdot[O]/dT
  J[94] -= dqdT; // dwdot[NO]/dT
  J[96] += dqdT; // dwdot[NO2]/dT

  // reaction 10: NO2 + M => NO + O + M
  // a third-body and non-pressure-fall-off reaction
  // Irreversible reaction in _ajac_reaction may not work
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[6];
  k_f = 10800000000 * exp(-(33011.0132767896) * invT);
  dlnkfdT = (33011.0132767896) * invT2;
  // rate of progress
  q_nocor = k_f * phi_f;
  q = alpha * q_nocor;
  dqdT = alpha * dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += q; // O
  wdot[4] += q; // NO
  wdot[6] -= q; // NO2
  // for convenience
  k_f *= alpha;
  if (consP == 1) {
    // d()/d[N2]
    dqdci = (1 - 1) * q_nocor;
    J[2] += dqdci; // dwdot[O]/d[N2]
    J[4] += dqdci; // dwdot[NO]/d[N2]
    J[6] -= dqdci; // dwdot[NO2]/d[N2]
    // d()/d[N]
    dqdci = (1 - 1) * q_nocor;
    J[12] += dqdci; // dwdot[O]/d[N]
    J[14] += dqdci; // dwdot[NO]/d[N]
    J[16] -= dqdci; // dwdot[NO2]/d[N]
    // d()/d[O]
    dqdci = (1 - 1) * q_nocor;
    J[22] += dqdci; // dwdot[O]/d[O]
    J[24] += dqdci; // dwdot[NO]/d[O]
    J[26] -= dqdci; // dwdot[NO2]/d[O]
    // d()/d[O2]
    dqdci = (1 - 1) * q_nocor;
    J[32] += dqdci; // dwdot[O]/d[O2]
    J[34] += dqdci; // dwdot[NO]/d[O2]
    J[36] -= dqdci; // dwdot[NO2]/d[O2]
    // d()/d[NO]
    dqdci = (1 - 1) * q_nocor;
    J[42] += dqdci; // dwdot[O]/d[NO]
    J[44] += dqdci; // dwdot[NO]/d[NO]
    J[46] -= dqdci; // dwdot[NO2]/d[NO]
    // d()/d[N2O]
    dqdci = (1 - 1) * q_nocor;
    J[52] += dqdci; // dwdot[O]/d[N2O]
    J[54] += dqdci; // dwdot[NO]/d[N2O]
    J[56] -= dqdci; // dwdot[NO2]/d[N2O]
    // d()/d[NO2]
    dqdci = (1 - 1) * q_nocor + k_f;
    J[62] += dqdci; // dwdot[O]/d[NO2]
    J[64] += dqdci; // dwdot[NO]/d[NO2]
    J[66] -= dqdci; // dwdot[NO2]/d[NO2]
    // d()/d[N2O3]
    dqdci = (1 - 1) * q_nocor;
    J[72] += dqdci; // dwdot[O]/d[N2O3]
    J[74] += dqdci; // dwdot[NO]/d[N2O3]
    J[76] -= dqdci; // dwdot[NO2]/d[N2O3]
    // d()/d[NO3]
    dqdci = (1 - 1) * q_nocor;
    J[82] += dqdci; // dwdot[O]/d[NO3]
    J[84] += dqdci; // dwdot[NO]/d[NO3]
    J[86] -= dqdci; // dwdot[NO2]/d[NO3]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor;
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor;
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor + k_f;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    for (int k = 0; k < 9; k++) {
      J[10 * k + 2] += dqdc[k];
      J[10 * k + 4] += dqdc[k];
      J[10 * k + 6] -= dqdc[k];
    }
  }
  J[92] += dqdT; // dwdot[O]/dT
  J[94] += dqdT; // dwdot[NO]/dT
  J[96] -= dqdT; // dwdot[NO2]/dT

  // reaction 16: N2O3 + M => NO + NO2 + M
  // a third-body and non-pressure-fall-off reaction
  // Irreversible reaction in _ajac_reaction may not work
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[7];
  k_f = 65000000 * exp(-(2626.79099550064) * invT);
  dlnkfdT = (2626.79099550064) * invT2;
  // rate of progress
  q_nocor = k_f * phi_f;
  q = alpha * q_nocor;
  dqdT = alpha * dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[4] += q; // NO
  wdot[6] += q; // NO2
  wdot[7] -= q; // N2O3
  // for convenience
  k_f *= alpha;
  if (consP == 1) {
    // d()/d[N2]
    dqdci = (1 - 1) * q_nocor;
    J[4] += dqdci; // dwdot[NO]/d[N2]
    J[6] += dqdci; // dwdot[NO2]/d[N2]
    J[7] -= dqdci; // dwdot[N2O3]/d[N2]
    // d()/d[N]
    dqdci = (1 - 1) * q_nocor;
    J[14] += dqdci; // dwdot[NO]/d[N]
    J[16] += dqdci; // dwdot[NO2]/d[N]
    J[17] -= dqdci; // dwdot[N2O3]/d[N]
    // d()/d[O]
    dqdci = (1 - 1) * q_nocor;
    J[24] += dqdci; // dwdot[NO]/d[O]
    J[26] += dqdci; // dwdot[NO2]/d[O]
    J[27] -= dqdci; // dwdot[N2O3]/d[O]
    // d()/d[O2]
    dqdci = (1 - 1) * q_nocor;
    J[34] += dqdci; // dwdot[NO]/d[O2]
    J[36] += dqdci; // dwdot[NO2]/d[O2]
    J[37] -= dqdci; // dwdot[N2O3]/d[O2]
    // d()/d[NO]
    dqdci = (1 - 1) * q_nocor;
    J[44] += dqdci; // dwdot[NO]/d[NO]
    J[46] += dqdci; // dwdot[NO2]/d[NO]
    J[47] -= dqdci; // dwdot[N2O3]/d[NO]
    // d()/d[N2O]
    dqdci = (1 - 1) * q_nocor;
    J[54] += dqdci; // dwdot[NO]/d[N2O]
    J[56] += dqdci; // dwdot[NO2]/d[N2O]
    J[57] -= dqdci; // dwdot[N2O3]/d[N2O]
    // d()/d[NO2]
    dqdci = (1 - 1) * q_nocor;
    J[64] += dqdci; // dwdot[NO]/d[NO2]
    J[66] += dqdci; // dwdot[NO2]/d[NO2]
    J[67] -= dqdci; // dwdot[N2O3]/d[NO2]
    // d()/d[N2O3]
    dqdci = (1 - 1) * q_nocor + k_f;
    J[74] += dqdci; // dwdot[NO]/d[N2O3]
    J[76] += dqdci; // dwdot[NO2]/d[N2O3]
    J[77] -= dqdci; // dwdot[N2O3]/d[N2O3]
    // d()/d[NO3]
    dqdci = (1 - 1) * q_nocor;
    J[84] += dqdci; // dwdot[NO]/d[NO3]
    J[86] += dqdci; // dwdot[NO2]/d[NO3]
    J[87] -= dqdci; // dwdot[N2O3]/d[NO3]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor;
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor;
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor + k_f;
    dqdc[8] = q_nocor;
    for (int k = 0; k < 9; k++) {
      J[10 * k + 4] += dqdc[k];
      J[10 * k + 6] += dqdc[k];
      J[10 * k + 7] -= dqdc[k];
    }
  }
  J[94] += dqdT; // dwdot[NO]/dT
  J[96] += dqdT; // dwdot[NO2]/dT
  J[97] -= dqdT; // dwdot[N2O3]/dT

  // reaction 17: NO + NO2 + M => N2O3 + M
  // a third-body and non-pressure-fall-off reaction
  // Irreversible reaction in _ajac_reaction may not work
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[4] * sc[6];
  k_f = 1.33e+21 * exp(-7.7 * logT);
  dlnkfdT = -7.7 * invT;
  // rate of progress
  q_nocor = k_f * phi_f;
  q = alpha * q_nocor;
  dqdT = alpha * dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[4] -= q; // NO
  wdot[6] -= q; // NO2
  wdot[7] += q; // N2O3
  // for convenience
  k_f *= alpha;
  if (consP == 1) {
    // d()/d[N2]
    dqdci = (1 - 1) * q_nocor;
    J[4] -= dqdci; // dwdot[NO]/d[N2]
    J[6] -= dqdci; // dwdot[NO2]/d[N2]
    J[7] += dqdci; // dwdot[N2O3]/d[N2]
    // d()/d[N]
    dqdci = (1 - 1) * q_nocor;
    J[14] -= dqdci; // dwdot[NO]/d[N]
    J[16] -= dqdci; // dwdot[NO2]/d[N]
    J[17] += dqdci; // dwdot[N2O3]/d[N]
    // d()/d[O]
    dqdci = (1 - 1) * q_nocor;
    J[24] -= dqdci; // dwdot[NO]/d[O]
    J[26] -= dqdci; // dwdot[NO2]/d[O]
    J[27] += dqdci; // dwdot[N2O3]/d[O]
    // d()/d[O2]
    dqdci = (1 - 1) * q_nocor;
    J[34] -= dqdci; // dwdot[NO]/d[O2]
    J[36] -= dqdci; // dwdot[NO2]/d[O2]
    J[37] += dqdci; // dwdot[N2O3]/d[O2]
    // d()/d[NO]
    dqdci = (1 - 1) * q_nocor + k_f * sc[6];
    J[44] -= dqdci; // dwdot[NO]/d[NO]
    J[46] -= dqdci; // dwdot[NO2]/d[NO]
    J[47] += dqdci; // dwdot[N2O3]/d[NO]
    // d()/d[N2O]
    dqdci = (1 - 1) * q_nocor;
    J[54] -= dqdci; // dwdot[NO]/d[N2O]
    J[56] -= dqdci; // dwdot[NO2]/d[N2O]
    J[57] += dqdci; // dwdot[N2O3]/d[N2O]
    // d()/d[NO2]
    dqdci = (1 - 1) * q_nocor + k_f * sc[4];
    J[64] -= dqdci; // dwdot[NO]/d[NO2]
    J[66] -= dqdci; // dwdot[NO2]/d[NO2]
    J[67] += dqdci; // dwdot[N2O3]/d[NO2]
    // d()/d[N2O3]
    dqdci = (1 - 1) * q_nocor;
    J[74] -= dqdci; // dwdot[NO]/d[N2O3]
    J[76] -= dqdci; // dwdot[NO2]/d[N2O3]
    J[77] += dqdci; // dwdot[N2O3]/d[N2O3]
    // d()/d[NO3]
    dqdci = (1 - 1) * q_nocor;
    J[84] -= dqdci; // dwdot[NO]/d[NO3]
    J[86] -= dqdci; // dwdot[NO2]/d[NO3]
    J[87] += dqdci; // dwdot[N2O3]/d[NO3]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor;
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor + k_f * sc[6];
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor + k_f * sc[4];
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    for (int k = 0; k < 9; k++) {
      J[10 * k + 4] -= dqdc[k];
      J[10 * k + 6] -= dqdc[k];
      J[10 * k + 7] += dqdc[k];
    }
  }
  J[94] -= dqdT; // dwdot[NO]/dT
  J[96] -= dqdT; // dwdot[NO2]/dT
  J[97] += dqdT; // dwdot[N2O3]/dT

  // reaction 0: N2 + O <=> N + NO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[2];
  k_f = 181000000 * exp(-(38395.4316009001) * invT);
  dlnkfdT = (38395.4316009001) * invT2;
  // reverse
  phi_r = sc[1] * sc[4];
  Kc = exp(g_RT[0] - g_RT[1] + g_RT[2] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[2]) + (h_RT[1] + h_RT[4]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // N2
  wdot[1] += q; // N
  wdot[2] -= q; // O
  wdot[4] += q; // NO
  // d()/d[N2]
  dqdci = +k_f * sc[2];
  J[0] -= dqdci; // dwdot[N2]/d[N2]
  J[1] += dqdci; // dwdot[N]/d[N2]
  J[2] -= dqdci; // dwdot[O]/d[N2]
  J[4] += dqdci; // dwdot[NO]/d[N2]
  // d()/d[N]
  dqdci = -k_r * sc[4];
  J[10] -= dqdci; // dwdot[N2]/d[N]
  J[11] += dqdci; // dwdot[N]/d[N]
  J[12] -= dqdci; // dwdot[O]/d[N]
  J[14] += dqdci; // dwdot[NO]/d[N]
  // d()/d[O]
  dqdci = +k_f * sc[0];
  J[20] -= dqdci; // dwdot[N2]/d[O]
  J[21] += dqdci; // dwdot[N]/d[O]
  J[22] -= dqdci; // dwdot[O]/d[O]
  J[24] += dqdci; // dwdot[NO]/d[O]
  // d()/d[NO]
  dqdci = -k_r * sc[1];
  J[40] -= dqdci; // dwdot[N2]/d[NO]
  J[41] += dqdci; // dwdot[N]/d[NO]
  J[42] -= dqdci; // dwdot[O]/d[NO]
  J[44] += dqdci; // dwdot[NO]/d[NO]
  // d()/dT
  J[90] -= dqdT; // dwdot[N2]/dT
  J[91] += dqdT; // dwdot[N]/dT
  J[92] -= dqdT; // dwdot[O]/dT
  J[94] += dqdT; // dwdot[NO]/dT

  // reaction 1: NO + O <=> N + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[4];
  k_f = 687 * exp(1.13 * logT - (19222.8766337403) * invT);
  dlnkfdT = 1.13 * invT + (19222.8766337403) * invT2;
  // reverse
  phi_r = sc[1] * sc[3];
  Kc = exp(-g_RT[1] + g_RT[2] - g_RT[3] + g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[4]) + (h_RT[1] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // N
  wdot[2] -= q; // O
  wdot[3] += q; // O2
  wdot[4] -= q; // NO
  // d()/d[N]
  dqdci = -k_r * sc[3];
  J[11] += dqdci; // dwdot[N]/d[N]
  J[12] -= dqdci; // dwdot[O]/d[N]
  J[13] += dqdci; // dwdot[O2]/d[N]
  J[14] -= dqdci; // dwdot[NO]/d[N]
  // d()/d[O]
  dqdci = +k_f * sc[4];
  J[21] += dqdci; // dwdot[N]/d[O]
  J[22] -= dqdci; // dwdot[O]/d[O]
  J[23] += dqdci; // dwdot[O2]/d[O]
  J[24] -= dqdci; // dwdot[NO]/d[O]
  // d()/d[O2]
  dqdci = -k_r * sc[1];
  J[31] += dqdci; // dwdot[N]/d[O2]
  J[32] -= dqdci; // dwdot[O]/d[O2]
  J[33] += dqdci; // dwdot[O2]/d[O2]
  J[34] -= dqdci; // dwdot[NO]/d[O2]
  // d()/d[NO]
  dqdci = +k_f * sc[2];
  J[41] += dqdci; // dwdot[N]/d[NO]
  J[42] -= dqdci; // dwdot[O]/d[NO]
  J[43] += dqdci; // dwdot[O2]/d[NO]
  J[44] -= dqdci; // dwdot[NO]/d[NO]
  // d()/dT
  J[91] += dqdT; // dwdot[N]/dT
  J[92] -= dqdT; // dwdot[O]/dT
  J[93] += dqdT; // dwdot[O2]/dT
  J[94] -= dqdT; // dwdot[NO]/dT

  // reaction 6: N2O + O => 2 NO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[5];
  k_f = 90300000 * exp(-(13939.1016427907) * invT);
  dlnkfdT = (13939.1016427907) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] -= q;     // O
  wdot[4] += 2 * q; // NO
  wdot[5] -= q;     // N2O
  // d()/d[O]
  dqdci = +k_f * sc[5];
  J[22] -= dqdci;     // dwdot[O]/d[O]
  J[24] += 2 * dqdci; // dwdot[NO]/d[O]
  J[25] -= dqdci;     // dwdot[N2O]/d[O]
  // d()/d[N2O]
  dqdci = +k_f * sc[2];
  J[52] -= dqdci;     // dwdot[O]/d[N2O]
  J[54] += 2 * dqdci; // dwdot[NO]/d[N2O]
  J[55] -= dqdci;     // dwdot[N2O]/d[N2O]
  // d()/dT
  J[92] -= dqdT;     // dwdot[O]/dT
  J[94] += 2 * dqdT; // dwdot[NO]/dT
  J[95] -= dqdT;     // dwdot[N2O]/dT

  // reaction 7: 2 NO => N2O + O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[4] * sc[4]);
  k_f = 3610000 * exp(-(32910.3699436287) * invT);
  dlnkfdT = (32910.3699436287) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += q;     // O
  wdot[4] -= 2 * q; // NO
  wdot[5] += q;     // N2O
  // d()/d[NO]
  dqdci = +k_f * 2.000000 * sc[4];
  J[42] += dqdci;      // dwdot[O]/d[NO]
  J[44] += -2 * dqdci; // dwdot[NO]/d[NO]
  J[45] += dqdci;      // dwdot[N2O]/d[NO]
  // d()/dT
  J[92] += dqdT;      // dwdot[O]/dT
  J[94] += -2 * dqdT; // dwdot[NO]/dT
  J[95] += dqdT;      // dwdot[N2O]/dT

  // reaction 8: N2O + O <=> N2 + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[5];
  k_f = 3670000 * exp(-(986304.664977251) * invT);
  dlnkfdT = (986304.664977251) * invT2;
  // reverse
  phi_r = sc[0] * sc[3];
  Kc = exp(-g_RT[0] + g_RT[2] - g_RT[3] + g_RT[5]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[5]) + (h_RT[0] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // N2
  wdot[2] -= q; // O
  wdot[3] += q; // O2
  wdot[5] -= q; // N2O
  // d()/d[N2]
  dqdci = -k_r * sc[3];
  J[0] += dqdci; // dwdot[N2]/d[N2]
  J[2] -= dqdci; // dwdot[O]/d[N2]
  J[3] += dqdci; // dwdot[O2]/d[N2]
  J[5] -= dqdci; // dwdot[N2O]/d[N2]
  // d()/d[O]
  dqdci = +k_f * sc[5];
  J[20] += dqdci; // dwdot[N2]/d[O]
  J[22] -= dqdci; // dwdot[O]/d[O]
  J[23] += dqdci; // dwdot[O2]/d[O]
  J[25] -= dqdci; // dwdot[N2O]/d[O]
  // d()/d[O2]
  dqdci = -k_r * sc[0];
  J[30] += dqdci; // dwdot[N2]/d[O2]
  J[32] -= dqdci; // dwdot[O]/d[O2]
  J[33] += dqdci; // dwdot[O2]/d[O2]
  J[35] -= dqdci; // dwdot[N2O]/d[O2]
  // d()/d[N2O]
  dqdci = +k_f * sc[2];
  J[50] += dqdci; // dwdot[N2]/d[N2O]
  J[52] -= dqdci; // dwdot[O]/d[N2O]
  J[53] += dqdci; // dwdot[O2]/d[N2O]
  J[55] -= dqdci; // dwdot[N2O]/d[N2O]
  // d()/dT
  J[90] += dqdT; // dwdot[N2]/dT
  J[92] -= dqdT; // dwdot[O]/dT
  J[93] += dqdT; // dwdot[O2]/dT
  J[95] -= dqdT; // dwdot[N2O]/dT

  // reaction 11: NO2 + O => NO + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[6];
  k_f = 3910000 * exp(-(-119.765566461523) * invT);
  dlnkfdT = (-119.765566461523) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] -= q; // O
  wdot[3] += q; // O2
  wdot[4] += q; // NO
  wdot[6] -= q; // NO2
  // d()/d[O]
  dqdci = +k_f * sc[6];
  J[22] -= dqdci; // dwdot[O]/d[O]
  J[23] += dqdci; // dwdot[O2]/d[O]
  J[24] += dqdci; // dwdot[NO]/d[O]
  J[26] -= dqdci; // dwdot[NO2]/d[O]
  // d()/d[NO2]
  dqdci = +k_f * sc[2];
  J[62] -= dqdci; // dwdot[O]/d[NO2]
  J[63] += dqdci; // dwdot[O2]/d[NO2]
  J[64] += dqdci; // dwdot[NO]/d[NO2]
  J[66] -= dqdci; // dwdot[NO2]/d[NO2]
  // d()/dT
  J[92] -= dqdT; // dwdot[O]/dT
  J[93] += dqdT; // dwdot[O2]/dT
  J[94] += dqdT; // dwdot[NO]/dT
  J[96] -= dqdT; // dwdot[NO2]/dT

  // reaction 12: 2 NO <=> N2 + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[4] * sc[4]);
  k_f = 12600000 * exp(-(30293.6432814441) * invT);
  dlnkfdT = (30293.6432814441) * invT2;
  // reverse
  phi_r = sc[0] * sc[3];
  Kc = exp(-g_RT[0] - g_RT[3] + 2.000000 * g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[4]) + (h_RT[0] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q;     // N2
  wdot[3] += q;     // O2
  wdot[4] -= 2 * q; // NO
  // d()/d[N2]
  dqdci = -k_r * sc[3];
  J[0] += dqdci;      // dwdot[N2]/d[N2]
  J[3] += dqdci;      // dwdot[O2]/d[N2]
  J[4] += -2 * dqdci; // dwdot[NO]/d[N2]
  // d()/d[O2]
  dqdci = -k_r * sc[0];
  J[30] += dqdci;      // dwdot[N2]/d[O2]
  J[33] += dqdci;      // dwdot[O2]/d[O2]
  J[34] += -2 * dqdci; // dwdot[NO]/d[O2]
  // d()/d[NO]
  dqdci = +k_f * 2.000000 * sc[4];
  J[40] += dqdci;      // dwdot[N2]/d[NO]
  J[43] += dqdci;      // dwdot[O2]/d[NO]
  J[44] += -2 * dqdci; // dwdot[NO]/d[NO]
  // d()/dT
  J[90] += dqdT;      // dwdot[N2]/dT
  J[93] += dqdT;      // dwdot[O2]/dT
  J[94] += -2 * dqdT; // dwdot[NO]/dT

  // reaction 13: N2O + NO => N2 + NO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[5];
  k_f = 18500000 * exp(-(24909.2249573336) * invT);
  dlnkfdT = (24909.2249573336) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[0] += q; // N2
  wdot[4] -= q; // NO
  wdot[5] -= q; // N2O
  wdot[6] += q; // NO2
  // d()/d[NO]
  dqdci = +k_f * sc[5];
  J[40] += dqdci; // dwdot[N2]/d[NO]
  J[44] -= dqdci; // dwdot[NO]/d[NO]
  J[45] -= dqdci; // dwdot[N2O]/d[NO]
  J[46] += dqdci; // dwdot[NO2]/d[NO]
  // d()/d[N2O]
  dqdci = +k_f * sc[4];
  J[50] += dqdci; // dwdot[N2]/d[N2O]
  J[54] -= dqdci; // dwdot[NO]/d[N2O]
  J[55] -= dqdci; // dwdot[N2O]/d[N2O]
  J[56] += dqdci; // dwdot[NO2]/d[N2O]
  // d()/dT
  J[90] += dqdT; // dwdot[N2]/dT
  J[94] -= dqdT; // dwdot[NO]/dT
  J[95] -= dqdT; // dwdot[N2O]/dT
  J[96] += dqdT; // dwdot[NO2]/dT

  // reaction 14: 2 NO + O2 => 2 NO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * (sc[4] * sc[4]);
  k_f = 0.0012 * exp(-(-528.377499094956) * invT);
  dlnkfdT = (-528.377499094956) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[3] -= q;     // O2
  wdot[4] -= 2 * q; // NO
  wdot[6] += 2 * q; // NO2
  // d()/d[O2]
  dqdci = +k_f * sc[4] * sc[4];
  J[33] -= dqdci;      // dwdot[O2]/d[O2]
  J[34] += -2 * dqdci; // dwdot[NO]/d[O2]
  J[36] += 2 * dqdci;  // dwdot[NO2]/d[O2]
  // d()/d[NO]
  dqdci = +k_f * sc[3] * 2.000000 * sc[4];
  J[43] -= dqdci;      // dwdot[O2]/d[NO]
  J[44] += -2 * dqdci; // dwdot[NO]/d[NO]
  J[46] += 2 * dqdci;  // dwdot[NO2]/d[NO]
  // d()/dT
  J[93] -= dqdT;      // dwdot[O2]/dT
  J[94] += -2 * dqdT; // dwdot[NO]/dT
  J[96] += 2 * dqdT;  // dwdot[NO2]/dT

  // reaction 15: 2 NO2 => 2 NO + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[6] * sc[6]);
  k_f = 1630000 * exp(-(13385.5633104055) * invT);
  dlnkfdT = (13385.5633104055) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[3] += q;     // O2
  wdot[4] += 2 * q; // NO
  wdot[6] -= 2 * q; // NO2
  // d()/d[NO2]
  dqdci = +k_f * 2.000000 * sc[6];
  J[63] += dqdci;      // dwdot[O2]/d[NO2]
  J[64] += 2 * dqdci;  // dwdot[NO]/d[NO2]
  J[66] += -2 * dqdci; // dwdot[NO2]/d[NO2]
  // d()/dT
  J[93] += dqdT;      // dwdot[O2]/dT
  J[94] += 2 * dqdT;  // dwdot[NO]/dT
  J[96] += -2 * dqdT; // dwdot[NO2]/dT

  // reaction 18: 2 NO2 => NO + NO3
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[6] * sc[6]);
  k_f = 9640 * exp(0.73 * logT - (10517.2283153186) * invT);
  dlnkfdT = 0.73 * invT + (10517.2283153186) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[4] += q;     // NO
  wdot[6] -= 2 * q; // NO2
  wdot[8] += q;     // NO3
  // d()/d[NO2]
  dqdci = +k_f * 2.000000 * sc[6];
  J[64] += dqdci;      // dwdot[NO]/d[NO2]
  J[66] += -2 * dqdci; // dwdot[NO2]/d[NO2]
  J[68] += dqdci;      // dwdot[NO3]/d[NO2]
  // d()/dT
  J[94] += dqdT;      // dwdot[NO]/dT
  J[96] += -2 * dqdT; // dwdot[NO2]/dT
  J[98] += dqdT;      // dwdot[NO3]/dT

  // reaction 19: NO + NO3 => 2 NO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[8];
  k_f = 10800000 * exp(-(-108.191583148015) * invT);
  dlnkfdT = (-108.191583148015) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[4] -= q;     // NO
  wdot[6] += 2 * q; // NO2
  wdot[8] -= q;     // NO3
  // d()/d[NO]
  dqdci = +k_f * sc[8];
  J[44] -= dqdci;     // dwdot[NO]/d[NO]
  J[46] += 2 * dqdci; // dwdot[NO2]/d[NO]
  J[48] -= dqdci;     // dwdot[NO3]/d[NO]
  // d()/d[NO3]
  dqdci = +k_f * sc[4];
  J[84] -= dqdci;     // dwdot[NO]/d[NO3]
  J[86] += 2 * dqdci; // dwdot[NO2]/d[NO3]
  J[88] -= dqdci;     // dwdot[NO3]/d[NO3]
  // d()/dT
  J[94] -= dqdT;     // dwdot[NO]/dT
  J[96] += 2 * dqdT; // dwdot[NO2]/dT
  J[98] -= dqdT;     // dwdot[NO3]/dT

  amrex::Real c_R[9], dcRdT[9], e_RT[9];
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
  for (int k = 0; k < 9; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[90 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 9; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 9; ++m) {
      dehmixdc += eh_RT[m] * J[k * 10 + m];
    }
    J[k * 10 + 9] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[99] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
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
  amrex::Real c[9];

  for (int k = 0; k < 9; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian(J, c, *Tp, *consP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 9; k++) {
    J[90 + k] *= 1.e-6;
    J[k * 10 + 9] *= 1.e6;
  }
}

// Transport function declarations

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENIMC(int* LENIMC)
{
  *LENIMC = 36;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENRMC(int* LENRMC)
{
  *LENRMC = 1782;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNO(int* NO)
{
  *NO = 4;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKK(int* KK)
{
  *KK = 9;
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
  WT[2] = 1.59990000E+01;
  WT[3] = 3.19980000E+01;
  WT[4] = 3.00060000E+01;
  WT[5] = 4.40130000E+01;
  WT[6] = 4.60050000E+01;
  WT[7] = 7.60110000E+01;
  WT[8] = 6.20040000E+01;
}

// the lennard-jones potential well depth eps/kb in K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetEPS(amrex::Real* EPS)
{
  EPS[0] = 9.75300000E+01;
  EPS[1] = 7.14000000E+01;
  EPS[2] = 8.00000000E+01;
  EPS[3] = 1.07400000E+02;
  EPS[4] = 9.75300000E+01;
  EPS[5] = 2.32400000E+02;
  EPS[6] = 2.00000000E+02;
  EPS[7] = 2.00000000E+02;
  EPS[8] = 2.00000000E+02;
}

// the lennard-jones collision diameter in Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetSIG(amrex::Real* SIG)
{
  SIG[0] = 3.62100000E+00;
  SIG[1] = 3.29800000E+00;
  SIG[2] = 2.75000000E+00;
  SIG[3] = 3.45800000E+00;
  SIG[4] = 3.62100000E+00;
  SIG[5] = 3.82800000E+00;
  SIG[6] = 3.50000000E+00;
  SIG[7] = 3.50000000E+00;
  SIG[8] = 3.90000000E+00;
}

// the dipole moment in Debye
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetDIP(amrex::Real* DIP)
{
  DIP[0] = 0.00000000E+00;
  DIP[1] = 0.00000000E+00;
  DIP[2] = 0.00000000E+00;
  DIP[3] = 0.00000000E+00;
  DIP[4] = 1.67100000E+00;
  DIP[5] = 6.00000000E-01;
  DIP[6] = 2.60700000E+00;
  DIP[7] = 2.13000000E+00;
  DIP[8] = 0.00000000E+00;
}

// the polarizability in cubic Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPOL(amrex::Real* POL)
{
  POL[0] = 1.76000000E+00;
  POL[1] = 1.10000000E+00;
  POL[2] = 8.02000000E-01;
  POL[3] = 1.56200000E+00;
  POL[4] = 1.76000000E+00;
  POL[5] = 2.99800000E+00;
  POL[6] = 2.91000000E+00;
  POL[7] = 4.65300000E+00;
  POL[8] = 3.00000000E+00;
}

// the rotational relaxation collision number at 298 K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetZROT(amrex::Real* ZROT)
{
  ZROT[0] = 4.00000000E+00;
  ZROT[1] = 0.00000000E+00;
  ZROT[2] = 0.00000000E+00;
  ZROT[3] = 3.80000000E+00;
  ZROT[4] = 4.00000000E+00;
  ZROT[5] = 1.00000000E+00;
  ZROT[6] = 1.00000000E+00;
  ZROT[7] = 1.00000000E+00;
  ZROT[8] = 1.00000000E+00;
}

// 0: monoatomic, 1: linear, 2: nonlinear
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLIN(int* NLIN)
{
  NLIN[0] = 1;
  NLIN[1] = 0;
  NLIN[2] = 0;
  NLIN[3] = 1;
  NLIN[4] = 1;
  NLIN[5] = 1;
  NLIN[6] = 2;
  NLIN[7] = 2;
  NLIN[8] = 2;
}

// Poly fits for the viscosities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFETA(amrex::Real* COFETA)
{
  COFETA[0] = -1.61980106E+01;
  COFETA[1] = 2.18577454E+00;
  COFETA[2] = -1.99190099E-01;
  COFETA[3] = 8.55173507E-03;
  COFETA[4] = -1.49200059E+01;
  COFETA[5] = 1.65982700E+00;
  COFETA[6] = -1.33228638E-01;
  COFETA[7] = 5.80456600E-03;
  COFETA[8] = -1.48293045E+01;
  COFETA[9] = 1.77514807E+00;
  COFETA[10] = -1.46750632E-01;
  COFETA[11] = 6.32347691E-03;
  COFETA[12] = -1.66882218E+01;
  COFETA[13] = 2.43208669E+00;
  COFETA[14] = -2.31031334E-01;
  COFETA[15] = 9.92288903E-03;
  COFETA[16] = -2.19648885E+01;
  COFETA[17] = 4.18223633E+00;
  COFETA[18] = -4.28968613E-01;
  COFETA[19] = 1.73906125E-02;
  COFETA[20] = -2.18026934E+01;
  COFETA[21] = 4.20248711E+00;
  COFETA[22] = -4.41648375E-01;
  COFETA[23] = 1.82712946E-02;
  COFETA[24] = -1.86755046E+01;
  COFETA[25] = 2.21591170E+00;
  COFETA[26] = -9.58450638E-02;
  COFETA[27] = 3.14090854E-04;
  COFETA[28] = -2.11886030E+01;
  COFETA[29] = 3.66480753E+00;
  COFETA[30] = -3.20676520E-01;
  COFETA[31] = 1.12348112E-02;
  COFETA[32] = -2.09722320E+01;
  COFETA[33] = 3.99889869E+00;
  COFETA[34] = -4.22064081E-01;
  COFETA[35] = 1.76869124E-02;
}

// Poly fits for the conductivities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFLAM(amrex::Real* COFLAM)
{
  COFLAM[0] = 5.57974408E+00;
  COFLAM[1] = -3.08343479E-01;
  COFLAM[2] = 1.78629219E-01;
  COFLAM[3] = -9.52248300E-03;
  COFLAM[4] = 1.99828576E+00;
  COFLAM[5] = 1.65982700E+00;
  COFLAM[6] = -1.33228638E-01;
  COFLAM[7] = 5.80456600E-03;
  COFLAM[8] = 1.95601814E+00;
  COFLAM[9] = 1.77514807E+00;
  COFLAM[10] = -1.46750632E-01;
  COFLAM[11] = 6.32347691E-03;
  COFLAM[12] = 1.32384755E-01;
  COFLAM[13] = 2.06559781E+00;
  COFLAM[14] = -1.62731262E-01;
  COFLAM[15] = 6.85619387E-03;
  COFLAM[16] = -1.74023448E+00;
  COFLAM[17] = 2.37924869E+00;
  COFLAM[18] = -1.53019536E-01;
  COFLAM[19] = 4.17985175E-03;
  COFLAM[20] = -1.08829479E+01;
  COFLAM[21] = 5.84868561E+00;
  COFLAM[22] = -5.85556154E-01;
  COFLAM[23] = 2.19649712E-02;
  COFLAM[24] = -1.24208651E+01;
  COFLAM[25] = 6.05280767E+00;
  COFLAM[26] = -5.84474072E-01;
  COFLAM[27] = 2.18940862E-02;
  COFLAM[28] = -9.71026407E+00;
  COFLAM[29] = 5.29559019E+00;
  COFLAM[30] = -4.89317854E-01;
  COFLAM[31] = 1.70984076E-02;
  COFLAM[32] = -2.02121447E+01;
  COFLAM[33] = 9.68301189E+00;
  COFLAM[34] = -1.09855350E+00;
  COFLAM[35] = 4.45899032E-02;
}

// Poly fits for the diffusion coefficients, dim NO*KK*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFD(amrex::Real* COFD)
{
  COFD[0] = -1.46523907E+01;
  COFD[1] = 3.08143963E+00;
  COFD[2] = -1.82377255E-01;
  COFD[3] = 7.74645838E-03;
  COFD[4] = -1.36573902E+01;
  COFD[5] = 2.82664829E+00;
  COFD[6] = -1.50396283E-01;
  COFD[7] = 6.40764097E-03;
  COFD[8] = -1.37959234E+01;
  COFD[9] = 2.92157208E+00;
  COFD[10] = -1.62325668E-01;
  COFD[11] = 6.90681066E-03;
  COFD[12] = -1.48890249E+01;
  COFD[13] = 3.17402131E+00;
  COFD[14] = -1.94181586E-01;
  COFD[15] = 8.24913005E-03;
  COFD[16] = -1.51227631E+01;
  COFD[17] = 3.24559156E+00;
  COFD[18] = -2.03401245E-01;
  COFD[19] = 8.64535815E-03;
  COFD[20] = -1.74329980E+01;
  COFD[21] = 4.03396152E+00;
  COFD[22] = -3.00949196E-01;
  COFD[23] = 1.26738860E-02;
  COFD[24] = -1.77343182E+01;
  COFD[25] = 4.15734303E+00;
  COFD[26] = -3.15484822E-01;
  COFD[27] = 1.32447425E-02;
  COFD[28] = -1.75536937E+01;
  COFD[29] = 4.07079478E+00;
  COFD[30] = -3.05346696E-01;
  COFD[31] = 1.28490195E-02;
  COFD[32] = -1.70159623E+01;
  COFD[33] = 3.85876343E+00;
  COFD[34] = -2.79731724E-01;
  COFD[35] = 1.18166515E-02;
  COFD[36] = -1.36573902E+01;
  COFD[37] = 2.82664829E+00;
  COFD[38] = -1.50396283E-01;
  COFD[39] = 6.40764097E-03;
  COFD[40] = -1.30372038E+01;
  COFD[41] = 2.70807453E+00;
  COFD[42] = -1.37739569E-01;
  COFD[43] = 5.98112990E-03;
  COFD[44] = -1.30387964E+01;
  COFD[45] = 2.75419603E+00;
  COFD[46] = -1.42892228E-01;
  COFD[47] = 6.16664401E-03;
  COFD[48] = -1.38541802E+01;
  COFD[49] = 2.90828671E+00;
  COFD[50] = -1.60674476E-01;
  COFD[51] = 6.83840384E-03;
  COFD[52] = -1.40291606E+01;
  COFD[53] = 2.95327314E+00;
  COFD[54] = -1.66292568E-01;
  COFD[55] = 7.07236335E-03;
  COFD[56] = -1.61686366E+01;
  COFD[57] = 3.70174834E+00;
  COFD[58] = -2.60619268E-01;
  COFD[59] = 1.10405358E-02;
  COFD[60] = -1.64687049E+01;
  COFD[61] = 3.83224417E+00;
  COFD[62] = -2.76545124E-01;
  COFD[63] = 1.16889493E-02;
  COFD[64] = -1.62401832E+01;
  COFD[65] = 3.73669524E+00;
  COFD[66] = -2.64919416E-01;
  COFD[67] = 1.12170216E-02;
  COFD[68] = -1.57386147E+01;
  COFD[69] = 3.52608084E+00;
  COFD[70] = -2.38858769E-01;
  COFD[71] = 1.01415298E-02;
  COFD[72] = -1.37959234E+01;
  COFD[73] = 2.92157208E+00;
  COFD[74] = -1.62325668E-01;
  COFD[75] = 6.90681066E-03;
  COFD[76] = -1.30387964E+01;
  COFD[77] = 2.75419603E+00;
  COFD[78] = -1.42892228E-01;
  COFD[79] = 6.16664401E-03;
  COFD[80] = -1.29251912E+01;
  COFD[81] = 2.75556205E+00;
  COFD[82] = -1.41528678E-01;
  COFD[83] = 6.04017282E-03;
  COFD[84] = -1.39812305E+01;
  COFD[85] = 2.99911091E+00;
  COFD[86] = -1.72035408E-01;
  COFD[87] = 7.31245587E-03;
  COFD[88] = -1.42280684E+01;
  COFD[89] = 3.06839486E+00;
  COFD[90] = -1.80739956E-01;
  COFD[91] = 7.67777502E-03;
  COFD[92] = -1.64040887E+01;
  COFD[93] = 3.82498885E+00;
  COFD[94] = -2.75667661E-01;
  COFD[95] = 1.16535505E-02;
  COFD[96] = -1.68532401E+01;
  COFD[97] = 4.00839481E+00;
  COFD[98] = -2.97871736E-01;
  COFD[99] = 1.25503082E-02;
  COFD[100] = -1.65541869E+01;
  COFD[101] = 3.88623414E+00;
  COFD[102] = -2.83022182E-01;
  COFD[103] = 1.19480932E-02;
  COFD[104] = -1.59623450E+01;
  COFD[105] = 3.64472985E+00;
  COFD[106] = -2.53544782E-01;
  COFD[107] = 1.07478008E-02;
  COFD[108] = -1.48890249E+01;
  COFD[109] = 3.17402131E+00;
  COFD[110] = -1.94181586E-01;
  COFD[111] = 8.24913005E-03;
  COFD[112] = -1.38541802E+01;
  COFD[113] = 2.90828671E+00;
  COFD[114] = -1.60674476E-01;
  COFD[115] = 6.83840384E-03;
  COFD[116] = -1.39812305E+01;
  COFD[117] = 2.99911091E+00;
  COFD[118] = -1.72035408E-01;
  COFD[119] = 7.31245587E-03;
  COFD[120] = -1.51750814E+01;
  COFD[121] = 3.28612365E+00;
  COFD[122] = -2.08601363E-01;
  COFD[123] = 8.86791307E-03;
  COFD[124] = -1.54111183E+01;
  COFD[125] = 3.35855699E+00;
  COFD[126] = -2.17890705E-01;
  COFD[127] = 9.26527664E-03;
  COFD[128] = -1.76986565E+01;
  COFD[129] = 4.12775615E+00;
  COFD[130] = -3.12111369E-01;
  COFD[131] = 1.31170101E-02;
  COFD[132] = -1.79806797E+01;
  COFD[133] = 4.24398236E+00;
  COFD[134] = -3.25644062E-01;
  COFD[135] = 1.36417375E-02;
  COFD[136] = -1.77970859E+01;
  COFD[137] = 4.15364774E+00;
  COFD[138] = -3.15063753E-01;
  COFD[139] = 1.32288171E-02;
  COFD[140] = -1.73137903E+01;
  COFD[141] = 3.96416887E+00;
  COFD[142] = -2.92543980E-01;
  COFD[143] = 1.23362352E-02;
  COFD[144] = -1.51227631E+01;
  COFD[145] = 3.24559156E+00;
  COFD[146] = -2.03401245E-01;
  COFD[147] = 8.64535815E-03;
  COFD[148] = -1.40291606E+01;
  COFD[149] = 2.95327314E+00;
  COFD[150] = -1.66292568E-01;
  COFD[151] = 7.07236335E-03;
  COFD[152] = -1.42280684E+01;
  COFD[153] = 3.06839486E+00;
  COFD[154] = -1.80739956E-01;
  COFD[155] = 7.67777502E-03;
  COFD[156] = -1.54111183E+01;
  COFD[157] = 3.35855699E+00;
  COFD[158] = -2.17890705E-01;
  COFD[159] = 9.26527664E-03;
  COFD[160] = -2.02465824E+01;
  COFD[161] = 5.01929964E+00;
  COFD[162] = -4.08539151E-01;
  COFD[163] = 1.65774362E-02;
  COFD[164] = -1.77827167E+01;
  COFD[165] = 4.13971449E+00;
  COFD[166] = -3.12121869E-01;
  COFD[167] = 1.30626034E-02;
  COFD[168] = -2.15908605E+01;
  COFD[169] = 5.22580041E+00;
  COFD[170] = -4.06299622E-01;
  COFD[171] = 1.54453722E-02;
  COFD[172] = -2.14527053E+01;
  COFD[173] = 5.26956560E+00;
  COFD[174] = -4.25771369E-01;
  COFD[175] = 1.67729394E-02;
  COFD[176] = -1.75165474E+01;
  COFD[177] = 4.02320601E+00;
  COFD[178] = -2.99655466E-01;
  COFD[179] = 1.26219718E-02;
  COFD[180] = -1.74329980E+01;
  COFD[181] = 4.03396152E+00;
  COFD[182] = -3.00949196E-01;
  COFD[183] = 1.26738860E-02;
  COFD[184] = -1.61686366E+01;
  COFD[185] = 3.70174834E+00;
  COFD[186] = -2.60619268E-01;
  COFD[187] = 1.10405358E-02;
  COFD[188] = -1.64040887E+01;
  COFD[189] = 3.82498885E+00;
  COFD[190] = -2.75667661E-01;
  COFD[191] = 1.16535505E-02;
  COFD[192] = -1.76986565E+01;
  COFD[193] = 4.12775615E+00;
  COFD[194] = -3.12111369E-01;
  COFD[195] = 1.31170101E-02;
  COFD[196] = -1.77827167E+01;
  COFD[197] = 4.13971449E+00;
  COFD[198] = -3.12121869E-01;
  COFD[199] = 1.30626034E-02;
  COFD[200] = -1.97753602E+01;
  COFD[201] = 4.73261665E+00;
  COFD[202] = -3.78086095E-01;
  COFD[203] = 1.54922530E-02;
  COFD[204] = -1.96291052E+01;
  COFD[205] = 4.68852609E+00;
  COFD[206] = -3.70144196E-01;
  COFD[207] = 1.50722475E-02;
  COFD[208] = -1.96177778E+01;
  COFD[209] = 4.65846218E+00;
  COFD[210] = -3.68002637E-01;
  COFD[211] = 1.50439179E-02;
  COFD[212] = -1.96147730E+01;
  COFD[213] = 4.66307592E+00;
  COFD[214] = -3.72086260E-01;
  COFD[215] = 1.53455397E-02;
  COFD[216] = -1.77343182E+01;
  COFD[217] = 4.15734303E+00;
  COFD[218] = -3.15484822E-01;
  COFD[219] = 1.32447425E-02;
  COFD[220] = -1.64687049E+01;
  COFD[221] = 3.83224417E+00;
  COFD[222] = -2.76545124E-01;
  COFD[223] = 1.16889493E-02;
  COFD[224] = -1.68532401E+01;
  COFD[225] = 4.00839481E+00;
  COFD[226] = -2.97871736E-01;
  COFD[227] = 1.25503082E-02;
  COFD[228] = -1.79806797E+01;
  COFD[229] = 4.24398236E+00;
  COFD[230] = -3.25644062E-01;
  COFD[231] = 1.36417375E-02;
  COFD[232] = -2.15908605E+01;
  COFD[233] = 5.22580041E+00;
  COFD[234] = -4.06299622E-01;
  COFD[235] = 1.54453722E-02;
  COFD[236] = -1.96291052E+01;
  COFD[237] = 4.68852609E+00;
  COFD[238] = -3.70144196E-01;
  COFD[239] = 1.50722475E-02;
  COFD[240] = -1.94158805E+01;
  COFD[241] = 3.90624281E+00;
  COFD[242] = -1.92686643E-01;
  COFD[243] = 4.78793438E-03;
  COFD[244] = -2.08782270E+01;
  COFD[245] = 4.62472212E+00;
  COFD[246] = -3.04768758E-01;
  COFD[247] = 1.02444464E-02;
  COFD[248] = -1.98403382E+01;
  COFD[249] = 4.74661795E+00;
  COFD[250] = -3.80575441E-01;
  COFD[251] = 1.56239689E-02;
  COFD[252] = -1.75536937E+01;
  COFD[253] = 4.07079478E+00;
  COFD[254] = -3.05346696E-01;
  COFD[255] = 1.28490195E-02;
  COFD[256] = -1.62401832E+01;
  COFD[257] = 3.73669524E+00;
  COFD[258] = -2.64919416E-01;
  COFD[259] = 1.12170216E-02;
  COFD[260] = -1.65541869E+01;
  COFD[261] = 3.88623414E+00;
  COFD[262] = -2.83022182E-01;
  COFD[263] = 1.19480932E-02;
  COFD[264] = -1.77970859E+01;
  COFD[265] = 4.15364774E+00;
  COFD[266] = -3.15063753E-01;
  COFD[267] = 1.32288171E-02;
  COFD[268] = -2.14527053E+01;
  COFD[269] = 5.26956560E+00;
  COFD[270] = -4.25771369E-01;
  COFD[271] = 1.67729394E-02;
  COFD[272] = -1.96177778E+01;
  COFD[273] = 4.65846218E+00;
  COFD[274] = -3.68002637E-01;
  COFD[275] = 1.50439179E-02;
  COFD[276] = -2.08782270E+01;
  COFD[277] = 4.62472212E+00;
  COFD[278] = -3.04768758E-01;
  COFD[279] = 1.02444464E-02;
  COFD[280] = -2.14925740E+01;
  COFD[281] = 4.96398988E+00;
  COFD[282] = -3.64597970E-01;
  COFD[283] = 1.33662096E-02;
  COFD[284] = -1.97555508E+01;
  COFD[285] = 4.68572807E+00;
  COFD[286] = -3.74542935E-01;
  COFD[287] = 1.54336267E-02;
  COFD[288] = -1.70159623E+01;
  COFD[289] = 3.85876343E+00;
  COFD[290] = -2.79731724E-01;
  COFD[291] = 1.18166515E-02;
  COFD[292] = -1.57386147E+01;
  COFD[293] = 3.52608084E+00;
  COFD[294] = -2.38858769E-01;
  COFD[295] = 1.01415298E-02;
  COFD[296] = -1.59623450E+01;
  COFD[297] = 3.64472985E+00;
  COFD[298] = -2.53544782E-01;
  COFD[299] = 1.07478008E-02;
  COFD[300] = -1.73137903E+01;
  COFD[301] = 3.96416887E+00;
  COFD[302] = -2.92543980E-01;
  COFD[303] = 1.23362352E-02;
  COFD[304] = -1.75165474E+01;
  COFD[305] = 4.02320601E+00;
  COFD[306] = -2.99655466E-01;
  COFD[307] = 1.26219718E-02;
  COFD[308] = -1.96147730E+01;
  COFD[309] = 4.66307592E+00;
  COFD[310] = -3.72086260E-01;
  COFD[311] = 1.53455397E-02;
  COFD[312] = -1.98403382E+01;
  COFD[313] = 4.74661795E+00;
  COFD[314] = -3.80575441E-01;
  COFD[315] = 1.56239689E-02;
  COFD[316] = -1.97555508E+01;
  COFD[317] = 4.68572807E+00;
  COFD[318] = -3.74542935E-01;
  COFD[319] = 1.54336267E-02;
  COFD[320] = -1.93199385E+01;
  COFD[321] = 4.53424564E+00;
  COFD[322] = -3.58127935E-01;
  COFD[323] = 1.48466859E-02;
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

  amrex::Real EPS[9];
  amrex::Real SIG[9];
  amrex::Real wt[9];
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

  // species 2: O
  Tci[2] = 1.316 * EPS[2];
  ai[2] = (5.55 * avogadro * avogadro * EPS[2] * boltzmann * 1e-24 * SIG[2] *
           SIG[2] * SIG[2]) /
          (wt[2] * wt[2]);
  bi[2] = 0.855 * avogadro * 1e-24 * SIG[2] * SIG[2] * SIG[2] / (wt[2]);
  acentric_i[2] = 0.0;

  // species 3: O2
  // Imported from NIST
  Tci[3] = 154.581000;
  ai[3] = 1e6 * 0.42748 * Rcst * Rcst * Tci[3] * Tci[3] /
          (31.998800 * 31.998800 * 50.430466);
  bi[3] = 0.08664 * Rcst * Tci[3] / (31.998800 * 50.430466);
  acentric_i[3] = 0.022200;

  // species 4: NO
  // Imported from NIST
  Tci[4] = 180.000000;
  ai[4] = 1e6 * 0.42748 * Rcst * Rcst * Tci[4] * Tci[4] /
          (30.006000 * 30.006000 * 64.800000);
  bi[4] = 0.08664 * Rcst * Tci[4] / (30.006000 * 64.800000);
  acentric_i[4] = 0.582000;

  // species 5: N2O
  // Imported from NIST
  Tci[5] = 309.600000;
  ai[5] = 1e6 * 0.42748 * Rcst * Rcst * Tci[5] * Tci[5] /
          (44.013000 * 44.013000 * 72.550000);
  bi[5] = 0.08664 * Rcst * Tci[5] / (44.013000 * 72.550000);
  acentric_i[5] = 0.162000;

  // species 6: NO2
  Tci[6] = 1.316 * EPS[6];
  ai[6] = (5.55 * avogadro * avogadro * EPS[6] * boltzmann * 1e-24 * SIG[6] *
           SIG[6] * SIG[6]) /
          (wt[6] * wt[6]);
  bi[6] = 0.855 * avogadro * 1e-24 * SIG[6] * SIG[6] * SIG[6] / (wt[6]);
  acentric_i[6] = 0.0;

  // species 7: N2O3
  Tci[7] = 1.316 * EPS[7];
  ai[7] = (5.55 * avogadro * avogadro * EPS[7] * boltzmann * 1e-24 * SIG[7] *
           SIG[7] * SIG[7]) /
          (wt[7] * wt[7]);
  bi[7] = 0.855 * avogadro * 1e-24 * SIG[7] * SIG[7] * SIG[7] / (wt[7]);
  acentric_i[7] = 0.0;

  // species 8: NO3
  Tci[8] = 1.316 * EPS[8];
  ai[8] = (5.55 * avogadro * avogadro * EPS[8] * boltzmann * 1e-24 * SIG[8] *
           SIG[8] * SIG[8]) /
          (wt[8] * wt[8]);
  bi[8] = 0.855 * avogadro * 1e-24 * SIG[8] * SIG[8] * SIG[8] / (wt[8]);
  acentric_i[8] = 0.0;
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

  // species 2: O
  // Computed from Lennard-Jones
  sqrtOneOverTc[2] = 9.7460146948181e-02;
  sqrtAsti[2] = 4.2500156012386e+04;
  Bi[2] = 6.6930234851885e-01;
  Fomega[2] = 4.8508000000000e-01;

  // species 3: O2
  // Imported from NIST
  sqrtOneOverTc[3] = 8.0430717653241e-02;
  sqrtAsti[3] = 3.6980081924740e+04;
  Bi[3] = 6.9005216444999e-01;
  Fomega[3] = 5.1945301904908e-01;

  // species 4: NO
  // Imported from NIST
  sqrtOneOverTc[4] = 7.4535599249993e-02;
  sqrtAsti[4] = 4.0510614916903e+04;
  Bi[4] = 6.6687106751983e-01;
  Fomega[4] = 1.3368144381880e+00;

  // species 5: N2O
  // Imported from NIST
  sqrtOneOverTc[5] = 5.6832861622539e-02;
  sqrtAsti[5] = 4.4894507333531e+04;
  Bi[5] = 6.9844955486412e-01;
  Fomega[5] = 7.3247646842800e-01;

  // species 6: NO2
  // Computed from Lennard-Jones
  sqrtOneOverTc[6] = 6.1639209090192e-02;
  sqrtAsti[6] = 3.3554545279404e+04;
  Bi[6] = 4.7986182770822e-01;
  Fomega[6] = 4.8508000000000e-01;

  // species 7: N2O3
  // Computed from Lennard-Jones
  sqrtOneOverTc[7] = 6.1639209090192e-02;
  sqrtAsti[7] = 2.0308598171041e+04;
  Bi[7] = 2.9043221880671e-01;
  Fomega[7] = 4.8508000000000e-01;

  // species 8: NO3
  // Computed from Lennard-Jones
  sqrtOneOverTc[8] = 6.1639209090192e-02;
  sqrtAsti[8] = 2.9284076265451e+04;
  Bi[8] = 4.9259638378590e-01;
  Fomega[8] = 4.8508000000000e-01;
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
