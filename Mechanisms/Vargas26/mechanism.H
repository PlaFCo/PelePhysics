#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/* Elements
0  Ar
1  H
2  C
*/

// Species
#define Ar_ID 0
#define H2_ID 1
#define H_ID 2
#define CH3_ID 3
#define CH4_ID 4
#define C2H_ID 5
#define C2H2_ID 6
#define C2H3_ID 7
#define C2H4_ID 8
#define C2H5_ID 9
#define C2H6_ID 10
#define C6H6_ID 11

#define NUM_GAS_ELEMENTS 3   // Elements in the homogeneous phase
#define NUM_GAS_SPECIES 12   // Species in the homogeneous phase
#define NUM_GAS_REACTIONS 10 // Reactions in the homogeneous phase

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
  kk = 12;
  ii = 10;
  nfit = -1; // Why do you need this anyway ?
}

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[12] = {
  2.5031289111389236e-02, // Ar
  4.9603174603174605e-01, // H2
  9.9206349206349209e-01, // H
  6.6511473229132020e-02, // CH3
  6.2332481456086768e-02, // CH4
  3.9952057530962849e-02, // C2H
  3.8405407481373381e-02, // C2H2
  3.6974044220956889e-02, // C2H3
  3.5645540742853074e-02, // C2H4
  3.4409194136673323e-02, // C2H5
  3.3255736614566010e-02, // C2H6
  1.2801802493791125e-02, // C6H6
};
#endif
const amrex::Real h_global_imw[12] = {
  2.5031289111389236e-02, // Ar
  4.9603174603174605e-01, // H2
  9.9206349206349209e-01, // H
  6.6511473229132020e-02, // CH3
  6.2332481456086768e-02, // CH4
  3.9952057530962849e-02, // C2H
  3.8405407481373381e-02, // C2H2
  3.6974044220956889e-02, // C2H3
  3.5645540742853074e-02, // C2H4
  3.4409194136673323e-02, // C2H5
  3.3255736614566010e-02, // C2H6
  1.2801802493791125e-02, // C6H6
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[12] = {
  3.9950000000000003e+01, // Ar
  2.0160000000000000e+00, // H2
  1.0080000000000000e+00, // H
  1.5035000000000000e+01, // CH3
  1.6042999999999999e+01, // CH4
  2.5029999999999998e+01, // C2H
  2.6037999999999997e+01, // C2H2
  2.7045999999999999e+01, // C2H3
  2.8053999999999998e+01, // C2H4
  2.9061999999999998e+01, // C2H5
  3.0070000000000000e+01, // C2H6
  7.8114000000000004e+01, // C6H6
};
#endif
const amrex::Real h_global_mw[12] = {
  3.9950000000000003e+01, // Ar
  2.0160000000000000e+00, // H2
  1.0080000000000000e+00, // H
  1.5035000000000000e+01, // CH3
  1.6042999999999999e+01, // CH4
  2.5029999999999998e+01, // C2H
  2.6037999999999997e+01, // C2H2
  2.7045999999999999e+01, // C2H3
  2.8053999999999998e+01, // C2H4
  2.9061999999999998e+01, // C2H5
  3.0070000000000000e+01, // C2H6
  7.8114000000000004e+01, // C6H6
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_imw(amrex::Real* imw_new)
{
  imw_new[0] = 2.5031289111389236e-02;  // Ar
  imw_new[1] = 4.9603174603174605e-01;  // H2
  imw_new[2] = 9.9206349206349209e-01;  // H
  imw_new[3] = 6.6511473229132020e-02;  // CH3
  imw_new[4] = 6.2332481456086768e-02;  // CH4
  imw_new[5] = 3.9952057530962849e-02;  // C2H
  imw_new[6] = 3.8405407481373381e-02;  // C2H2
  imw_new[7] = 3.6974044220956889e-02;  // C2H3
  imw_new[8] = 3.5645540742853074e-02;  // C2H4
  imw_new[9] = 3.4409194136673323e-02;  // C2H5
  imw_new[10] = 3.3255736614566010e-02; // C2H6
  imw_new[11] = 1.2801802493791125e-02; // C6H6
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
  mw_new[0] = 3.9950000000000003e+01;  // Ar
  mw_new[1] = 2.0160000000000000e+00;  // H2
  mw_new[2] = 1.0080000000000000e+00;  // H
  mw_new[3] = 1.5035000000000000e+01;  // CH3
  mw_new[4] = 1.6042999999999999e+01;  // CH4
  mw_new[5] = 2.5029999999999998e+01;  // C2H
  mw_new[6] = 2.6037999999999997e+01;  // C2H2
  mw_new[7] = 2.7045999999999999e+01;  // C2H3
  mw_new[8] = 2.8053999999999998e+01;  // C2H4
  mw_new[9] = 2.9061999999999998e+01;  // C2H5
  mw_new[10] = 3.0070000000000000e+01; // C2H6
  mw_new[11] = 7.8114000000000004e+01; // C6H6
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
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with no change across T
  // species 0: Ar
  species[0] = +1.50000000e+00;
  // species 2: H
  species[2] = +1.50000000e+00;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: H2
    species[1] = +2.50207268e+00 + 8.65475654e-05 * T - 2.63683344e-07 * T2 +
                 3.37306621e-10 * T3 - 2.92359706e-14 * T4;
    // species 3: CH3
    species[3] = +2.65717970e+00 + 2.12659790e-03 * T + 5.45838830e-06 * T2 -
                 6.61810030e-09 * T3 + 2.46570740e-12 * T4;
    // species 4: CH4
    species[4] = +4.14825732e+00 - 1.37002410e-02 * T + 4.93749414e-05 * T2 -
                 4.91952339e-08 * T3 + 1.70097299e-11 * T4;
    // species 5: C2H
    species[5] = +1.89867676e+00 + 1.32988489e-02 * T - 2.80733327e-05 * T2 +
                 2.89484755e-08 * T3 - 1.07502351e-11 * T4;
    // species 6: C2H2
    species[6] = -1.91320318e-01 + 2.33615762e-02 * T - 3.55172234e-05 * T2 +
                 2.80152958e-08 * T3 - 8.50075165e-12 * T4;
    // species 7: C2H3
    species[7] = +2.36377642e+00 + 2.65765722e-04 * T + 2.79620704e-05 * T2 -
                 3.72986942e-08 * T3 + 1.51590176e-11 * T4;
    // species 8: C2H4
    species[8] = +2.95920063e+00 - 7.57051373e-03 * T + 5.70989993e-05 * T2 -
                 6.91588352e-08 * T3 + 2.69884190e-11 * T4;
    // species 9: C2H5
    species[9] = +3.24185905e+00 - 3.56905235e-03 * T + 4.82667202e-05 * T2 -
                 5.85401009e-08 * T3 + 2.25804514e-11 * T4;
    // species 10: C2H6
    species[10] = +3.29142572e+00 - 5.50154901e-03 * T + 5.99438458e-05 * T2 -
                  7.08466469e-08 * T3 + 2.68685836e-11 * T4;
    // species 11: C6H6
    species[11] = -4.95181368e-01 + 1.85020642e-02 * T + 7.38345881e-05 * T2 -
                  1.18135741e-07 * T3 + 5.07210429e-11 * T4;
  } else {
    // species 1: H2
    species[1] = +1.98711895e+00 + 7.36069465e-04 * T - 9.00982609e-08 * T2 -
                 7.41224720e-13 * T3 + 6.58618037e-16 * T4;
    // species 3: CH3
    species[3] = +1.97812060e+00 + 5.79785200e-03 * T - 1.97558000e-06 * T2 +
                 3.07297900e-10 * T3 - 1.79174160e-14 * T4;
    // species 4: CH4
    species[4] = +9.11786000e-01 + 9.60267960e-03 * T - 3.38387841e-06 * T2 +
                 5.38797240e-10 * T3 - 3.19306807e-14 * T4;
    // species 5: C2H
    species[5] = +2.66270248e+00 + 3.82492252e-03 * T - 1.36632500e-06 * T2 +
                 2.13455040e-10 * T3 - 1.23216848e-14 * T4;
    // species 6: C2H2
    species[6] = +3.65878489e+00 + 4.88396667e-03 * T - 1.60828888e-06 * T2 +
                 2.46974544e-10 * T3 - 1.38605959e-14 * T4;
    // species 7: C2H3
    species[7] = +3.15026763e+00 + 7.54021341e-03 * T - 2.62997847e-06 * T2 +
                 4.15974048e-10 * T3 - 2.45407509e-14 * T4;
    // species 8: C2H4
    species[8] = +2.99182724e+00 + 1.04833908e-02 * T - 3.71721342e-06 * T2 +
                 5.94628366e-10 * T3 - 3.53630386e-14 * T4;
    // species 9: C2H5
    species[9] = +3.32195633e+00 + 1.23930542e-02 * T - 4.39680960e-06 * T2 +
                 7.03519917e-10 * T3 - 4.18435239e-14 * T4;
    // species 10: C2H6
    species[10] = +3.04666411e+00 + 1.53538802e-02 * T - 5.47039485e-06 * T2 +
                  8.77826544e-10 * T3 - 5.23167531e-14 * T4;
    // species 11: C6H6
    species[11] = +1.00809576e+01 + 2.07176746e-02 * T - 7.52145991e-06 * T2 +
                  1.22320984e-09 * T3 - 7.36091279e-14 * T4;
  }
}

// compute Cp/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cp_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with no change across T
  // species 0: Ar
  species[0] = +2.50000000e+00;
  // species 2: H
  species[2] = +2.50000000e+00;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: H2
    species[1] = +3.50207268e+00 + 8.65475654e-05 * T - 2.63683344e-07 * T2 +
                 3.37306621e-10 * T3 - 2.92359706e-14 * T4;
    // species 3: CH3
    species[3] = +3.65717970e+00 + 2.12659790e-03 * T + 5.45838830e-06 * T2 -
                 6.61810030e-09 * T3 + 2.46570740e-12 * T4;
    // species 4: CH4
    species[4] = +5.14825732e+00 - 1.37002410e-02 * T + 4.93749414e-05 * T2 -
                 4.91952339e-08 * T3 + 1.70097299e-11 * T4;
    // species 5: C2H
    species[5] = +2.89867676e+00 + 1.32988489e-02 * T - 2.80733327e-05 * T2 +
                 2.89484755e-08 * T3 - 1.07502351e-11 * T4;
    // species 6: C2H2
    species[6] = +8.08679682e-01 + 2.33615762e-02 * T - 3.55172234e-05 * T2 +
                 2.80152958e-08 * T3 - 8.50075165e-12 * T4;
    // species 7: C2H3
    species[7] = +3.36377642e+00 + 2.65765722e-04 * T + 2.79620704e-05 * T2 -
                 3.72986942e-08 * T3 + 1.51590176e-11 * T4;
    // species 8: C2H4
    species[8] = +3.95920063e+00 - 7.57051373e-03 * T + 5.70989993e-05 * T2 -
                 6.91588352e-08 * T3 + 2.69884190e-11 * T4;
    // species 9: C2H5
    species[9] = +4.24185905e+00 - 3.56905235e-03 * T + 4.82667202e-05 * T2 -
                 5.85401009e-08 * T3 + 2.25804514e-11 * T4;
    // species 10: C2H6
    species[10] = +4.29142572e+00 - 5.50154901e-03 * T + 5.99438458e-05 * T2 -
                  7.08466469e-08 * T3 + 2.68685836e-11 * T4;
    // species 11: C6H6
    species[11] = +5.04818632e-01 + 1.85020642e-02 * T + 7.38345881e-05 * T2 -
                  1.18135741e-07 * T3 + 5.07210429e-11 * T4;
  } else {
    // species 1: H2
    species[1] = +2.98711895e+00 + 7.36069465e-04 * T - 9.00982609e-08 * T2 -
                 7.41224720e-13 * T3 + 6.58618037e-16 * T4;
    // species 3: CH3
    species[3] = +2.97812060e+00 + 5.79785200e-03 * T - 1.97558000e-06 * T2 +
                 3.07297900e-10 * T3 - 1.79174160e-14 * T4;
    // species 4: CH4
    species[4] = +1.91178600e+00 + 9.60267960e-03 * T - 3.38387841e-06 * T2 +
                 5.38797240e-10 * T3 - 3.19306807e-14 * T4;
    // species 5: C2H
    species[5] = +3.66270248e+00 + 3.82492252e-03 * T - 1.36632500e-06 * T2 +
                 2.13455040e-10 * T3 - 1.23216848e-14 * T4;
    // species 6: C2H2
    species[6] = +4.65878489e+00 + 4.88396667e-03 * T - 1.60828888e-06 * T2 +
                 2.46974544e-10 * T3 - 1.38605959e-14 * T4;
    // species 7: C2H3
    species[7] = +4.15026763e+00 + 7.54021341e-03 * T - 2.62997847e-06 * T2 +
                 4.15974048e-10 * T3 - 2.45407509e-14 * T4;
    // species 8: C2H4
    species[8] = +3.99182724e+00 + 1.04833908e-02 * T - 3.71721342e-06 * T2 +
                 5.94628366e-10 * T3 - 3.53630386e-14 * T4;
    // species 9: C2H5
    species[9] = +4.32195633e+00 + 1.23930542e-02 * T - 4.39680960e-06 * T2 +
                 7.03519917e-10 * T3 - 4.18435239e-14 * T4;
    // species 10: C2H6
    species[10] = +4.04666411e+00 + 1.53538802e-02 * T - 5.47039485e-06 * T2 +
                  8.77826544e-10 * T3 - 5.23167531e-14 * T4;
    // species 11: C6H6
    species[11] = +1.10809576e+01 + 2.07176746e-02 * T - 7.52145991e-06 * T2 +
                  1.22320984e-09 * T3 - 7.36091279e-14 * T4;
  }
}

// compute the g/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
gibbs(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 0: Ar
  species[0] = -7.453750000000000e+02 * invT - 1.879674910000000e+00 -
               2.500000000000000e+00 * logT;
  // species 2: H
  species[2] = +2.547366000000000e+04 * invT + 2.946682850000000e+00 -
               2.500000000000000e+00 * logT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: H2
    species[1] = -1.046312790000000e+03 * invT + 7.760830270000000e+00 -
                 3.502072680000000e+00 * logT - 4.327378270000000e-05 * T +
                 4.394722400000000e-08 * T2 - 2.810888508333333e-11 * T3 +
                 1.461798530000000e-15 * T4;
    // species 3: CH3
    species[3] = +1.642271600000000e+04 * invT + 1.983644300000000e+00 -
                 3.657179700000000e+00 * logT - 1.063298950000000e-03 * T -
                 9.097313833333333e-07 * T2 + 5.515083583333334e-10 * T3 -
                 1.232853700000000e-13 * T4;
    // species 4: CH4
    species[4] = -1.024532220000000e+04 * invT + 9.781484580000001e+00 -
                 5.148257320000000e+00 * logT + 6.850120500000000e-03 * T -
                 8.229156900000000e-06 * T2 + 4.099602825000000e-09 * T3 -
                 8.504864950000000e-13 * T4;
    // species 5: C2H
    species[5] = +6.706160500000000e+04 * invT - 3.286799560000000e+00 -
                 2.898676760000000e+00 * logT - 6.649424450000000e-03 * T +
                 4.678888783333334e-06 * T2 - 2.412372958333333e-09 * T3 +
                 5.375117550000000e-13 * T4;
    // species 6: C2H2
    species[6] = +2.642898080000000e+04 * invT - 1.313099641800000e+01 -
                 8.086796820000000e-01 * logT - 1.168078810000000e-02 * T +
                 5.919537233333333e-06 * T2 - 2.334607983333333e-09 * T3 +
                 4.250375825000001e-13 * T4;
    // species 7: C2H3
    species[7] = +3.447495890000000e+04 * invT - 4.551324500000000e+00 -
                 3.363776420000000e+00 * logT - 1.328828610000000e-04 * T -
                 4.660345066666666e-06 * T2 + 3.108224516666667e-09 * T3 -
                 7.579508800000001e-13 * T4;
    // species 8: C2H4
    species[8] = +5.089775980000000e+03 * invT - 1.381014999999999e-01 -
                 3.959200630000000e+00 * logT + 3.785256865000000e-03 * T -
                 9.516499883333334e-06 * T2 + 5.763236266666667e-09 * T3 -
                 1.349420950000000e-12 * T4;
    // species 9: C2H5
    species[9] = +1.296903440000000e+04 * invT - 2.051787699999998e-01 -
                 4.241859050000000e+00 * logT + 1.784526175000000e-03 * T -
                 8.044453366666667e-06 * T2 + 4.878341741666667e-09 * T3 -
                 1.129022570000000e-12 * T4;
    // species 10: C2H6
    species[10] = -1.152220560000000e+04 * invT + 1.624635780000000e+00 -
                  4.291425720000000e+00 * logT + 2.750774505000000e-03 * T -
                  9.990640966666668e-06 * T2 + 5.903887241666666e-09 * T3 -
                  1.343429180000000e-12 * T4;
    // species 11: C6H6
    species[11] = +8.552479130000000e+03 * invT - 2.113647066800000e+01 -
                  5.048186320000000e-01 * logT - 9.251032100000001e-03 * T -
                  1.230576468333333e-05 * T2 + 9.844645083333332e-09 * T3 -
                  2.536052145000000e-12 * T4;
  } else {
    // species 1: H2
    species[1] = -8.354486590000000e+02 * invT + 4.319807720000000e+00 -
                 2.987118950000000e+00 * logT - 3.680347325000000e-04 * T +
                 1.501637681666667e-08 * T2 + 6.176872666666667e-14 * T3 -
                 3.293090185000000e-17 * T4;
    // species 3: CH3
    species[3] = +1.650951300000000e+04 * invT - 1.744359300000000e+00 -
                 2.978120600000000e+00 * logT - 2.898926000000000e-03 * T +
                 3.292633333333333e-07 * T2 - 2.560815833333334e-11 * T3 +
                 8.958708000000000e-16 * T4;
    // species 4: CH4
    species[4] = -1.009921360000000e+04 * invT - 6.570632610000000e+00 -
                 1.911786000000000e+00 * logT - 4.801339800000000e-03 * T +
                 5.639797350000000e-07 * T2 - 4.489977000000000e-11 * T3 +
                 1.596534035000000e-15 * T4;
    // species 5: C2H
    species[5] = +6.716837900000000e+04 * invT - 2.593554399999998e-01 -
                 3.662702480000000e+00 * logT - 1.912461260000000e-03 * T +
                 2.277208333333333e-07 * T2 - 1.778792000000000e-11 * T3 +
                 6.160842400000000e-16 * T4;
    // species 6: C2H2
    species[6] = +2.575940420000000e+04 * invT + 8.657166830000000e+00 -
                 4.658784890000000e+00 * logT - 2.441983335000000e-03 * T +
                 2.680481466666667e-07 * T2 - 2.058121200000000e-11 * T3 +
                 6.930297950000000e-16 * T4;
    // species 7: C2H3
    species[7] = +3.385663800000000e+04 * invT + 2.422145280000000e+00 -
                 4.150267630000000e+00 * logT - 3.770106705000000e-03 * T +
                 4.383297450000000e-07 * T2 - 3.466450400000000e-11 * T3 +
                 1.227037545000000e-15 * T4;
    // species 8: C2H4
    species[8] = +4.268658510000000e+03 * invT + 4.260909002000000e+00 -
                 3.991827240000000e+00 * logT - 5.241695400000000e-03 * T +
                 6.195355700000000e-07 * T2 - 4.955236383333334e-11 * T3 +
                 1.768151930000000e-15 * T4;
    // species 9: C2H5
    species[9] = +1.217594750000000e+04 * invT + 4.150852521000000e+00 -
                 4.321956330000000e+00 * logT - 6.196527100000000e-03 * T +
                 7.328016000000000e-07 * T2 - 5.862665974999999e-11 * T3 +
                 2.092176195000000e-15 * T4;
    // species 10: C2H6
    species[10] = -1.244734990000000e+04 * invT + 5.015362423000000e+00 -
                  4.046664110000000e+00 * logT - 7.676940100000000e-03 * T +
                  9.117324750000000e-07 * T2 - 7.315221200000000e-11 * T3 +
                  2.615837655000000e-15 * T4;
    // species 11: C6H6
    species[11] = +4.306410350000000e+03 * invT + 5.112228860000000e+01 -
                  1.108095760000000e+01 * logT - 1.035883730000000e-02 * T +
                  1.253576651666667e-06 * T2 - 1.019341533333333e-10 * T3 +
                  3.680456395000000e-15 * T4;
  }
}

// compute the a/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
helmholtz(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 0: Ar
  species[0] = -7.45375000e+02 * invT - 2.87967491e+00 - 2.50000000e+00 * logT;
  // species 2: H
  species[2] = +2.54736600e+04 * invT + 1.94668285e+00 - 2.50000000e+00 * logT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: H2
    species[1] = -1.04631279e+03 * invT + 6.76083027e+00 -
                 3.50207268e+00 * logT - 4.32737827e-05 * T +
                 4.39472240e-08 * T2 - 2.81088851e-11 * T3 +
                 1.46179853e-15 * T4;
    // species 3: CH3
    species[3] = +1.64227160e+04 * invT + 9.83644300e-01 -
                 3.65717970e+00 * logT - 1.06329895e-03 * T -
                 9.09731383e-07 * T2 + 5.51508358e-10 * T3 -
                 1.23285370e-13 * T4;
    // species 4: CH4
    species[4] = -1.02453222e+04 * invT + 8.78148458e+00 -
                 5.14825732e+00 * logT + 6.85012050e-03 * T -
                 8.22915690e-06 * T2 + 4.09960282e-09 * T3 -
                 8.50486495e-13 * T4;
    // species 5: C2H
    species[5] = +6.70616050e+04 * invT - 4.28679956e+00 -
                 2.89867676e+00 * logT - 6.64942445e-03 * T +
                 4.67888878e-06 * T2 - 2.41237296e-09 * T3 +
                 5.37511755e-13 * T4;
    // species 6: C2H2
    species[6] = +2.64289808e+04 * invT - 1.41309964e+01 -
                 8.08679682e-01 * logT - 1.16807881e-02 * T +
                 5.91953723e-06 * T2 - 2.33460798e-09 * T3 +
                 4.25037583e-13 * T4;
    // species 7: C2H3
    species[7] = +3.44749589e+04 * invT - 5.55132450e+00 -
                 3.36377642e+00 * logT - 1.32882861e-04 * T -
                 4.66034507e-06 * T2 + 3.10822452e-09 * T3 -
                 7.57950880e-13 * T4;
    // species 8: C2H4
    species[8] = +5.08977598e+03 * invT - 1.13810150e+00 -
                 3.95920063e+00 * logT + 3.78525686e-03 * T -
                 9.51649988e-06 * T2 + 5.76323627e-09 * T3 -
                 1.34942095e-12 * T4;
    // species 9: C2H5
    species[9] = +1.29690344e+04 * invT - 1.20517877e+00 -
                 4.24185905e+00 * logT + 1.78452617e-03 * T -
                 8.04445337e-06 * T2 + 4.87834174e-09 * T3 -
                 1.12902257e-12 * T4;
    // species 10: C2H6
    species[10] = -1.15222056e+04 * invT + 6.24635780e-01 -
                  4.29142572e+00 * logT + 2.75077450e-03 * T -
                  9.99064097e-06 * T2 + 5.90388724e-09 * T3 -
                  1.34342918e-12 * T4;
    // species 11: C6H6
    species[11] = +8.55247913e+03 * invT - 2.21364707e+01 -
                  5.04818632e-01 * logT - 9.25103210e-03 * T -
                  1.23057647e-05 * T2 + 9.84464508e-09 * T3 -
                  2.53605215e-12 * T4;
  } else {
    // species 1: H2
    species[1] = -8.35448659e+02 * invT + 3.31980772e+00 -
                 2.98711895e+00 * logT - 3.68034732e-04 * T +
                 1.50163768e-08 * T2 + 6.17687267e-14 * T3 -
                 3.29309019e-17 * T4;
    // species 3: CH3
    species[3] = +1.65095130e+04 * invT - 2.74435930e+00 -
                 2.97812060e+00 * logT - 2.89892600e-03 * T +
                 3.29263333e-07 * T2 - 2.56081583e-11 * T3 +
                 8.95870800e-16 * T4;
    // species 4: CH4
    species[4] = -1.00992136e+04 * invT - 7.57063261e+00 -
                 1.91178600e+00 * logT - 4.80133980e-03 * T +
                 5.63979735e-07 * T2 - 4.48997700e-11 * T3 +
                 1.59653404e-15 * T4;
    // species 5: C2H
    species[5] = +6.71683790e+04 * invT - 1.25935544e+00 -
                 3.66270248e+00 * logT - 1.91246126e-03 * T +
                 2.27720833e-07 * T2 - 1.77879200e-11 * T3 +
                 6.16084240e-16 * T4;
    // species 6: C2H2
    species[6] = +2.57594042e+04 * invT + 7.65716683e+00 -
                 4.65878489e+00 * logT - 2.44198333e-03 * T +
                 2.68048147e-07 * T2 - 2.05812120e-11 * T3 +
                 6.93029795e-16 * T4;
    // species 7: C2H3
    species[7] = +3.38566380e+04 * invT + 1.42214528e+00 -
                 4.15026763e+00 * logT - 3.77010670e-03 * T +
                 4.38329745e-07 * T2 - 3.46645040e-11 * T3 +
                 1.22703754e-15 * T4;
    // species 8: C2H4
    species[8] = +4.26865851e+03 * invT + 3.26090900e+00 -
                 3.99182724e+00 * logT - 5.24169540e-03 * T +
                 6.19535570e-07 * T2 - 4.95523638e-11 * T3 +
                 1.76815193e-15 * T4;
    // species 9: C2H5
    species[9] = +1.21759475e+04 * invT + 3.15085252e+00 -
                 4.32195633e+00 * logT - 6.19652710e-03 * T +
                 7.32801600e-07 * T2 - 5.86266597e-11 * T3 +
                 2.09217619e-15 * T4;
    // species 10: C2H6
    species[10] = -1.24473499e+04 * invT + 4.01536242e+00 -
                  4.04666411e+00 * logT - 7.67694010e-03 * T +
                  9.11732475e-07 * T2 - 7.31522120e-11 * T3 +
                  2.61583765e-15 * T4;
    // species 11: C6H6
    species[11] = +4.30641035e+03 * invT + 5.01222886e+01 -
                  1.10809576e+01 * logT - 1.03588373e-02 * T +
                  1.25357665e-06 * T2 - 1.01934153e-10 * T3 +
                  3.68045639e-15 * T4;
  }
}

// compute the e/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesInternalEnergy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 0: Ar
  species[0] = +1.50000000e+00 - 7.45375000e+02 * invT;
  // species 2: H
  species[2] = +1.50000000e+00 + 2.54736600e+04 * invT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: H2
    species[1] = +2.50207268e+00 + 4.32737827e-05 * T - 8.78944480e-08 * T2 +
                 8.43266552e-11 * T3 - 5.84719412e-15 * T4 -
                 1.04631279e+03 * invT;
    // species 3: CH3
    species[3] = +2.65717970e+00 + 1.06329895e-03 * T + 1.81946277e-06 * T2 -
                 1.65452507e-09 * T3 + 4.93141480e-13 * T4 +
                 1.64227160e+04 * invT;
    // species 4: CH4
    species[4] = +4.14825732e+00 - 6.85012050e-03 * T + 1.64583138e-05 * T2 -
                 1.22988085e-08 * T3 + 3.40194598e-12 * T4 -
                 1.02453222e+04 * invT;
    // species 5: C2H
    species[5] = +1.89867676e+00 + 6.64942445e-03 * T - 9.35777757e-06 * T2 +
                 7.23711887e-09 * T3 - 2.15004702e-12 * T4 +
                 6.70616050e+04 * invT;
    // species 6: C2H2
    species[6] = -1.91320318e-01 + 1.16807881e-02 * T - 1.18390745e-05 * T2 +
                 7.00382395e-09 * T3 - 1.70015033e-12 * T4 +
                 2.64289808e+04 * invT;
    // species 7: C2H3
    species[7] = +2.36377642e+00 + 1.32882861e-04 * T + 9.32069013e-06 * T2 -
                 9.32467355e-09 * T3 + 3.03180352e-12 * T4 +
                 3.44749589e+04 * invT;
    // species 8: C2H4
    species[8] = +2.95920063e+00 - 3.78525686e-03 * T + 1.90329998e-05 * T2 -
                 1.72897088e-08 * T3 + 5.39768380e-12 * T4 +
                 5.08977598e+03 * invT;
    // species 9: C2H5
    species[9] = +3.24185905e+00 - 1.78452617e-03 * T + 1.60889067e-05 * T2 -
                 1.46350252e-08 * T3 + 4.51609028e-12 * T4 +
                 1.29690344e+04 * invT;
    // species 10: C2H6
    species[10] = +3.29142572e+00 - 2.75077450e-03 * T + 1.99812819e-05 * T2 -
                  1.77116617e-08 * T3 + 5.37371672e-12 * T4 -
                  1.15222056e+04 * invT;
    // species 11: C6H6
    species[11] = -4.95181368e-01 + 9.25103210e-03 * T + 2.46115294e-05 * T2 -
                  2.95339352e-08 * T3 + 1.01442086e-11 * T4 +
                  8.55247913e+03 * invT;
  } else {
    // species 1: H2
    species[1] = +1.98711895e+00 + 3.68034732e-04 * T - 3.00327536e-08 * T2 -
                 1.85306180e-13 * T3 + 1.31723607e-16 * T4 -
                 8.35448659e+02 * invT;
    // species 3: CH3
    species[3] = +1.97812060e+00 + 2.89892600e-03 * T - 6.58526667e-07 * T2 +
                 7.68244750e-11 * T3 - 3.58348320e-15 * T4 +
                 1.65095130e+04 * invT;
    // species 4: CH4
    species[4] = +9.11786000e-01 + 4.80133980e-03 * T - 1.12795947e-06 * T2 +
                 1.34699310e-10 * T3 - 6.38613614e-15 * T4 -
                 1.00992136e+04 * invT;
    // species 5: C2H
    species[5] = +2.66270248e+00 + 1.91246126e-03 * T - 4.55441667e-07 * T2 +
                 5.33637600e-11 * T3 - 2.46433696e-15 * T4 +
                 6.71683790e+04 * invT;
    // species 6: C2H2
    species[6] = +3.65878489e+00 + 2.44198333e-03 * T - 5.36096293e-07 * T2 +
                 6.17436360e-11 * T3 - 2.77211918e-15 * T4 +
                 2.57594042e+04 * invT;
    // species 7: C2H3
    species[7] = +3.15026763e+00 + 3.77010670e-03 * T - 8.76659490e-07 * T2 +
                 1.03993512e-10 * T3 - 4.90815018e-15 * T4 +
                 3.38566380e+04 * invT;
    // species 8: C2H4
    species[8] = +2.99182724e+00 + 5.24169540e-03 * T - 1.23907114e-06 * T2 +
                 1.48657092e-10 * T3 - 7.07260772e-15 * T4 +
                 4.26865851e+03 * invT;
    // species 9: C2H5
    species[9] = +3.32195633e+00 + 6.19652710e-03 * T - 1.46560320e-06 * T2 +
                 1.75879979e-10 * T3 - 8.36870478e-15 * T4 +
                 1.21759475e+04 * invT;
    // species 10: C2H6
    species[10] = +3.04666411e+00 + 7.67694010e-03 * T - 1.82346495e-06 * T2 +
                  2.19456636e-10 * T3 - 1.04633506e-14 * T4 -
                  1.24473499e+04 * invT;
    // species 11: C6H6
    species[11] = +1.00809576e+01 + 1.03588373e-02 * T - 2.50715330e-06 * T2 +
                  3.05802460e-10 * T3 - 1.47218256e-14 * T4 +
                  4.30641035e+03 * invT;
  }
}

// compute the h/(RT) at the given temperature (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEnthalpy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 0: Ar
  species[0] = +2.50000000e+00 - 7.45375000e+02 * invT;
  // species 2: H
  species[2] = +2.50000000e+00 + 2.54736600e+04 * invT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: H2
    species[1] = +3.50207268e+00 + 4.32737827e-05 * T - 8.78944480e-08 * T2 +
                 8.43266552e-11 * T3 - 5.84719412e-15 * T4 -
                 1.04631279e+03 * invT;
    // species 3: CH3
    species[3] = +3.65717970e+00 + 1.06329895e-03 * T + 1.81946277e-06 * T2 -
                 1.65452507e-09 * T3 + 4.93141480e-13 * T4 +
                 1.64227160e+04 * invT;
    // species 4: CH4
    species[4] = +5.14825732e+00 - 6.85012050e-03 * T + 1.64583138e-05 * T2 -
                 1.22988085e-08 * T3 + 3.40194598e-12 * T4 -
                 1.02453222e+04 * invT;
    // species 5: C2H
    species[5] = +2.89867676e+00 + 6.64942445e-03 * T - 9.35777757e-06 * T2 +
                 7.23711887e-09 * T3 - 2.15004702e-12 * T4 +
                 6.70616050e+04 * invT;
    // species 6: C2H2
    species[6] = +8.08679682e-01 + 1.16807881e-02 * T - 1.18390745e-05 * T2 +
                 7.00382395e-09 * T3 - 1.70015033e-12 * T4 +
                 2.64289808e+04 * invT;
    // species 7: C2H3
    species[7] = +3.36377642e+00 + 1.32882861e-04 * T + 9.32069013e-06 * T2 -
                 9.32467355e-09 * T3 + 3.03180352e-12 * T4 +
                 3.44749589e+04 * invT;
    // species 8: C2H4
    species[8] = +3.95920063e+00 - 3.78525686e-03 * T + 1.90329998e-05 * T2 -
                 1.72897088e-08 * T3 + 5.39768380e-12 * T4 +
                 5.08977598e+03 * invT;
    // species 9: C2H5
    species[9] = +4.24185905e+00 - 1.78452617e-03 * T + 1.60889067e-05 * T2 -
                 1.46350252e-08 * T3 + 4.51609028e-12 * T4 +
                 1.29690344e+04 * invT;
    // species 10: C2H6
    species[10] = +4.29142572e+00 - 2.75077450e-03 * T + 1.99812819e-05 * T2 -
                  1.77116617e-08 * T3 + 5.37371672e-12 * T4 -
                  1.15222056e+04 * invT;
    // species 11: C6H6
    species[11] = +5.04818632e-01 + 9.25103210e-03 * T + 2.46115294e-05 * T2 -
                  2.95339352e-08 * T3 + 1.01442086e-11 * T4 +
                  8.55247913e+03 * invT;
  } else {
    // species 1: H2
    species[1] = +2.98711895e+00 + 3.68034732e-04 * T - 3.00327536e-08 * T2 -
                 1.85306180e-13 * T3 + 1.31723607e-16 * T4 -
                 8.35448659e+02 * invT;
    // species 3: CH3
    species[3] = +2.97812060e+00 + 2.89892600e-03 * T - 6.58526667e-07 * T2 +
                 7.68244750e-11 * T3 - 3.58348320e-15 * T4 +
                 1.65095130e+04 * invT;
    // species 4: CH4
    species[4] = +1.91178600e+00 + 4.80133980e-03 * T - 1.12795947e-06 * T2 +
                 1.34699310e-10 * T3 - 6.38613614e-15 * T4 -
                 1.00992136e+04 * invT;
    // species 5: C2H
    species[5] = +3.66270248e+00 + 1.91246126e-03 * T - 4.55441667e-07 * T2 +
                 5.33637600e-11 * T3 - 2.46433696e-15 * T4 +
                 6.71683790e+04 * invT;
    // species 6: C2H2
    species[6] = +4.65878489e+00 + 2.44198333e-03 * T - 5.36096293e-07 * T2 +
                 6.17436360e-11 * T3 - 2.77211918e-15 * T4 +
                 2.57594042e+04 * invT;
    // species 7: C2H3
    species[7] = +4.15026763e+00 + 3.77010670e-03 * T - 8.76659490e-07 * T2 +
                 1.03993512e-10 * T3 - 4.90815018e-15 * T4 +
                 3.38566380e+04 * invT;
    // species 8: C2H4
    species[8] = +3.99182724e+00 + 5.24169540e-03 * T - 1.23907114e-06 * T2 +
                 1.48657092e-10 * T3 - 7.07260772e-15 * T4 +
                 4.26865851e+03 * invT;
    // species 9: C2H5
    species[9] = +4.32195633e+00 + 6.19652710e-03 * T - 1.46560320e-06 * T2 +
                 1.75879979e-10 * T3 - 8.36870478e-15 * T4 +
                 1.21759475e+04 * invT;
    // species 10: C2H6
    species[10] = +4.04666411e+00 + 7.67694010e-03 * T - 1.82346495e-06 * T2 +
                  2.19456636e-10 * T3 - 1.04633506e-14 * T4 -
                  1.24473499e+04 * invT;
    // species 11: C6H6
    species[11] = +1.10809576e+01 + 1.03588373e-02 * T - 2.50715330e-06 * T2 +
                  3.05802460e-10 * T3 - 1.47218256e-14 * T4 +
                  4.30641035e+03 * invT;
  }
}

// compute the S/R at the given temperature (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEntropy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 0: Ar
  species[0] = +2.50000000e+00 * logT + 4.37967491e+00;
  // species 2: H
  species[2] = +2.50000000e+00 * logT - 4.46682850e-01;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: H2
    species[1] = +3.50207268e+00 * logT + 8.65475654e-05 * T -
                 1.31841672e-07 * T2 + 1.12435540e-10 * T3 -
                 7.30899265e-15 * T4 - 4.25875759e+00;
    // species 3: CH3
    species[3] = +3.65717970e+00 * logT + 2.12659790e-03 * T +
                 2.72919415e-06 * T2 - 2.20603343e-09 * T3 +
                 6.16426850e-13 * T4 + 1.67353540e+00;
    // species 4: CH4
    species[4] = +5.14825732e+00 * logT - 1.37002410e-02 * T +
                 2.46874707e-05 * T2 - 1.63984113e-08 * T3 +
                 4.25243248e-12 * T4 - 4.63322726e+00;
    // species 5: C2H
    species[5] = +2.89867676e+00 * logT + 1.32988489e-02 * T -
                 1.40366663e-05 * T2 + 9.64949183e-09 * T3 -
                 2.68755878e-12 * T4 + 6.18547632e+00;
    // species 6: C2H2
    species[6] = +8.08679682e-01 * logT + 2.33615762e-02 * T -
                 1.77586117e-05 * T2 + 9.33843193e-09 * T3 -
                 2.12518791e-12 * T4 + 1.39396761e+01;
    // species 7: C2H3
    species[7] = +3.36377642e+00 * logT + 2.65765722e-04 * T +
                 1.39810352e-05 * T2 - 1.24328981e-08 * T3 +
                 3.78975440e-12 * T4 + 7.91510092e+00;
    // species 8: C2H4
    species[8] = +3.95920063e+00 * logT - 7.57051373e-03 * T +
                 2.85494997e-05 * T2 - 2.30529451e-08 * T3 +
                 6.74710475e-12 * T4 + 4.09730213e+00;
    // species 9: C2H5
    species[9] = +4.24185905e+00 * logT - 3.56905235e-03 * T +
                 2.41333601e-05 * T2 - 1.95133670e-08 * T3 +
                 5.64511285e-12 * T4 + 4.44703782e+00;
    // species 10: C2H6
    species[10] = +4.29142572e+00 * logT - 5.50154901e-03 * T +
                  2.99719229e-05 * T2 - 2.36155490e-08 * T3 +
                  6.71714590e-12 * T4 + 2.66678994e+00;
    // species 11: C6H6
    species[11] = +5.04818632e-01 * logT + 1.85020642e-02 * T +
                  3.69172940e-05 * T2 - 3.93785803e-08 * T3 +
                  1.26802607e-11 * T4 + 2.16412893e+01;
  } else {
    // species 1: H2
    species[1] = +2.98711895e+00 * logT + 7.36069465e-04 * T -
                 4.50491305e-08 * T2 - 2.47074907e-13 * T3 +
                 1.64654509e-16 * T4 - 1.33268877e+00;
    // species 3: CH3
    species[3] = +2.97812060e+00 * logT + 5.79785200e-03 * T -
                 9.87790000e-07 * T2 + 1.02432633e-10 * T3 -
                 4.47935400e-15 * T4 + 4.72247990e+00;
    // species 4: CH4
    species[4] = +1.91178600e+00 * logT + 9.60267960e-03 * T -
                 1.69193921e-06 * T2 + 1.79599080e-10 * T3 -
                 7.98267018e-15 * T4 + 8.48241861e+00;
    // species 5: C2H
    species[5] = +3.66270248e+00 * logT + 3.82492252e-03 * T -
                 6.83162500e-07 * T2 + 7.11516800e-11 * T3 -
                 3.08042120e-15 * T4 + 3.92205792e+00;
    // species 6: C2H2
    species[6] = +4.65878489e+00 * logT + 4.88396667e-03 * T -
                 8.04144440e-07 * T2 + 8.23248480e-11 * T3 -
                 3.46514898e-15 * T4 - 3.99838194e+00;
    // species 7: C2H3
    species[7] = +4.15026763e+00 * logT + 7.54021341e-03 * T -
                 1.31498923e-06 * T2 + 1.38658016e-10 * T3 -
                 6.13518772e-15 * T4 + 1.72812235e+00;
    // species 8: C2H4
    species[8] = +3.99182724e+00 * logT + 1.04833908e-02 * T -
                 1.85860671e-06 * T2 + 1.98209455e-10 * T3 -
                 8.84075965e-15 * T4 - 2.69081762e-01;
    // species 9: C2H5
    species[9] = +4.32195633e+00 * logT + 1.23930542e-02 * T -
                 2.19840480e-06 * T2 + 2.34506639e-10 * T3 -
                 1.04608810e-14 * T4 + 1.71103809e-01;
    // species 10: C2H6
    species[10] = +4.04666411e+00 * logT + 1.53538802e-02 * T -
                  2.73519742e-06 * T2 + 2.92608848e-10 * T3 -
                  1.30791883e-14 * T4 - 9.68698313e-01;
    // species 11: C6H6
    species[11] = +1.10809576e+01 * logT + 2.07176746e-02 * T -
                  3.76072996e-06 * T2 + 4.07736613e-10 * T3 -
                  1.84022820e-14 * T4 - 4.00413310e+01;
  }
}

// compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
dcvpRdT(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;

  // species with no change across T
  // species 0: Ar
  species[0] = 0.0;
  // species 2: H
  species[2] = 0.0;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: H2
    species[1] = +8.65475654e-05 - 5.27366688e-07 * T + 1.01191986e-09 * T2 -
                 1.16943882e-13 * T3;
    // species 3: CH3
    species[3] = +2.12659790e-03 + 1.09167766e-05 * T - 1.98543009e-08 * T2 +
                 9.86282960e-12 * T3;
    // species 4: CH4
    species[4] = -1.37002410e-02 + 9.87498828e-05 * T - 1.47585702e-07 * T2 +
                 6.80389196e-11 * T3;
    // species 5: C2H
    species[5] = +1.32988489e-02 - 5.61466654e-05 * T + 8.68454265e-08 * T2 -
                 4.30009404e-11 * T3;
    // species 6: C2H2
    species[6] = +2.33615762e-02 - 7.10344468e-05 * T + 8.40458874e-08 * T2 -
                 3.40030066e-11 * T3;
    // species 7: C2H3
    species[7] = +2.65765722e-04 + 5.59241408e-05 * T - 1.11896083e-07 * T2 +
                 6.06360704e-11 * T3;
    // species 8: C2H4
    species[8] = -7.57051373e-03 + 1.14197999e-04 * T - 2.07476506e-07 * T2 +
                 1.07953676e-10 * T3;
    // species 9: C2H5
    species[9] = -3.56905235e-03 + 9.65334404e-05 * T - 1.75620303e-07 * T2 +
                 9.03218056e-11 * T3;
    // species 10: C2H6
    species[10] = -5.50154901e-03 + 1.19887692e-04 * T - 2.12539941e-07 * T2 +
                  1.07474334e-10 * T3;
    // species 11: C6H6
    species[11] = +1.85020642e-02 + 1.47669176e-04 * T - 3.54407223e-07 * T2 +
                  2.02884172e-10 * T3;
  } else {
    // species 1: H2
    species[1] = +7.36069465e-04 - 1.80196522e-07 * T - 2.22367416e-12 * T2 +
                 2.63447215e-15 * T3;
    // species 3: CH3
    species[3] = +5.79785200e-03 - 3.95116000e-06 * T + 9.21893700e-10 * T2 -
                 7.16696640e-14 * T3;
    // species 4: CH4
    species[4] = +9.60267960e-03 - 6.76775682e-06 * T + 1.61639172e-09 * T2 -
                 1.27722723e-13 * T3;
    // species 5: C2H
    species[5] = +3.82492252e-03 - 2.73265000e-06 * T + 6.40365120e-10 * T2 -
                 4.92867392e-14 * T3;
    // species 6: C2H2
    species[6] = +4.88396667e-03 - 3.21657776e-06 * T + 7.40923632e-10 * T2 -
                 5.54423836e-14 * T3;
    // species 7: C2H3
    species[7] = +7.54021341e-03 - 5.25995694e-06 * T + 1.24792214e-09 * T2 -
                 9.81630036e-14 * T3;
    // species 8: C2H4
    species[8] = +1.04833908e-02 - 7.43442684e-06 * T + 1.78388510e-09 * T2 -
                 1.41452154e-13 * T3;
    // species 9: C2H5
    species[9] = +1.23930542e-02 - 8.79361920e-06 * T + 2.11055975e-09 * T2 -
                 1.67374096e-13 * T3;
    // species 10: C2H6
    species[10] = +1.53538802e-02 - 1.09407897e-05 * T + 2.63347963e-09 * T2 -
                  2.09267012e-13 * T3;
    // species 11: C6H6
    species[11] = +2.07176746e-02 - 1.50429198e-05 * T + 3.66962952e-09 * T2 -
                  2.94436512e-13 * T3;
  }
}

// Returns the mean specific heat at CP (Eq. 33)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cpbl)
{
  amrex::Real result = 0;
  amrex::Real cpor[12]; //  temporary storage
  cp_R(cpor, T);

  // perform dot product
  for (int id = 0; id < 12; ++id) {
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
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with no change across T
  // species 0: Ar
  result += y[0] * (+2.50000000e+00) * 0.0250312891113892;
  // species 2: H
  result += y[2] * (+2.50000000e+00) * 0.9920634920634921;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: H2
    result += y[1] *
              (+3.50207268e+00 + 8.65475654e-05 * T - 2.63683344e-07 * T2 +
               3.37306621e-10 * T3 - 2.92359706e-14 * T4) *
              0.4960317460317460;
    // species 3: CH3
    result += y[3] *
              (+3.65717970e+00 + 2.12659790e-03 * T + 5.45838830e-06 * T2 -
               6.61810030e-09 * T3 + 2.46570740e-12 * T4) *
              0.0665114732291320;
    // species 4: CH4
    result += y[4] *
              (+5.14825732e+00 - 1.37002410e-02 * T + 4.93749414e-05 * T2 -
               4.91952339e-08 * T3 + 1.70097299e-11 * T4) *
              0.0623324814560868;
    // species 5: C2H
    result += y[5] *
              (+2.89867676e+00 + 1.32988489e-02 * T - 2.80733327e-05 * T2 +
               2.89484755e-08 * T3 - 1.07502351e-11 * T4) *
              0.0399520575309628;
    // species 6: C2H2
    result += y[6] *
              (+8.08679682e-01 + 2.33615762e-02 * T - 3.55172234e-05 * T2 +
               2.80152958e-08 * T3 - 8.50075165e-12 * T4) *
              0.0384054074813734;
    // species 7: C2H3
    result += y[7] *
              (+3.36377642e+00 + 2.65765722e-04 * T + 2.79620704e-05 * T2 -
               3.72986942e-08 * T3 + 1.51590176e-11 * T4) *
              0.0369740442209569;
    // species 8: C2H4
    result += y[8] *
              (+3.95920063e+00 - 7.57051373e-03 * T + 5.70989993e-05 * T2 -
               6.91588352e-08 * T3 + 2.69884190e-11 * T4) *
              0.0356455407428531;
    // species 9: C2H5
    result += y[9] *
              (+4.24185905e+00 - 3.56905235e-03 * T + 4.82667202e-05 * T2 -
               5.85401009e-08 * T3 + 2.25804514e-11 * T4) *
              0.0344091941366733;
    // species 10: C2H6
    result += y[10] *
              (+4.29142572e+00 - 5.50154901e-03 * T + 5.99438458e-05 * T2 -
               7.08466469e-08 * T3 + 2.68685836e-11 * T4) *
              0.0332557366145660;
    // species 11: C6H6
    result += y[11] *
              (+5.04818632e-01 + 1.85020642e-02 * T + 7.38345881e-05 * T2 -
               1.18135741e-07 * T3 + 5.07210429e-11 * T4) *
              0.0128018024937911;
  } else {
    // species 1: H2
    result += y[1] *
              (+2.98711895e+00 + 7.36069465e-04 * T - 9.00982609e-08 * T2 -
               7.41224720e-13 * T3 + 6.58618037e-16 * T4) *
              0.4960317460317460;
    // species 3: CH3
    result += y[3] *
              (+2.97812060e+00 + 5.79785200e-03 * T - 1.97558000e-06 * T2 +
               3.07297900e-10 * T3 - 1.79174160e-14 * T4) *
              0.0665114732291320;
    // species 4: CH4
    result += y[4] *
              (+1.91178600e+00 + 9.60267960e-03 * T - 3.38387841e-06 * T2 +
               5.38797240e-10 * T3 - 3.19306807e-14 * T4) *
              0.0623324814560868;
    // species 5: C2H
    result += y[5] *
              (+3.66270248e+00 + 3.82492252e-03 * T - 1.36632500e-06 * T2 +
               2.13455040e-10 * T3 - 1.23216848e-14 * T4) *
              0.0399520575309628;
    // species 6: C2H2
    result += y[6] *
              (+4.65878489e+00 + 4.88396667e-03 * T - 1.60828888e-06 * T2 +
               2.46974544e-10 * T3 - 1.38605959e-14 * T4) *
              0.0384054074813734;
    // species 7: C2H3
    result += y[7] *
              (+4.15026763e+00 + 7.54021341e-03 * T - 2.62997847e-06 * T2 +
               4.15974048e-10 * T3 - 2.45407509e-14 * T4) *
              0.0369740442209569;
    // species 8: C2H4
    result += y[8] *
              (+3.99182724e+00 + 1.04833908e-02 * T - 3.71721342e-06 * T2 +
               5.94628366e-10 * T3 - 3.53630386e-14 * T4) *
              0.0356455407428531;
    // species 9: C2H5
    result += y[9] *
              (+4.32195633e+00 + 1.23930542e-02 * T - 4.39680960e-06 * T2 +
               7.03519917e-10 * T3 - 4.18435239e-14 * T4) *
              0.0344091941366733;
    // species 10: C2H6
    result += y[10] *
              (+4.04666411e+00 + 1.53538802e-02 * T - 5.47039485e-06 * T2 +
               8.77826544e-10 * T3 - 5.23167531e-14 * T4) *
              0.0332557366145660;
    // species 11: C6H6
    result += y[11] *
              (+1.10809576e+01 + 2.07176746e-02 * T - 7.52145991e-06 * T2 +
               1.22320984e-09 * T3 - 7.36091279e-14 * T4) *
              0.0128018024937911;
  }

  cpbs = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 35)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cvbl)
{
  amrex::Real result = 0;
  amrex::Real cvor[12]; //  temporary storage
  cv_R(cvor, T);

  // perform dot product
  for (int id = 0; id < 12; ++id) {
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
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with no change across T
  // species 0: Ar
  result += y[0] * (+1.50000000e+00) * 0.0250312891113892;
  // species 2: H
  result += y[2] * (+1.50000000e+00) * 0.9920634920634921;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: H2
    result += y[1] *
              (+2.50207268e+00 + 8.65475654e-05 * T - 2.63683344e-07 * T2 +
               3.37306621e-10 * T3 - 2.92359706e-14 * T4) *
              0.4960317460317460;
    // species 3: CH3
    result += y[3] *
              (+2.65717970e+00 + 2.12659790e-03 * T + 5.45838830e-06 * T2 -
               6.61810030e-09 * T3 + 2.46570740e-12 * T4) *
              0.0665114732291320;
    // species 4: CH4
    result += y[4] *
              (+4.14825732e+00 - 1.37002410e-02 * T + 4.93749414e-05 * T2 -
               4.91952339e-08 * T3 + 1.70097299e-11 * T4) *
              0.0623324814560868;
    // species 5: C2H
    result += y[5] *
              (+1.89867676e+00 + 1.32988489e-02 * T - 2.80733327e-05 * T2 +
               2.89484755e-08 * T3 - 1.07502351e-11 * T4) *
              0.0399520575309628;
    // species 6: C2H2
    result += y[6] *
              (-1.91320318e-01 + 2.33615762e-02 * T - 3.55172234e-05 * T2 +
               2.80152958e-08 * T3 - 8.50075165e-12 * T4) *
              0.0384054074813734;
    // species 7: C2H3
    result += y[7] *
              (+2.36377642e+00 + 2.65765722e-04 * T + 2.79620704e-05 * T2 -
               3.72986942e-08 * T3 + 1.51590176e-11 * T4) *
              0.0369740442209569;
    // species 8: C2H4
    result += y[8] *
              (+2.95920063e+00 - 7.57051373e-03 * T + 5.70989993e-05 * T2 -
               6.91588352e-08 * T3 + 2.69884190e-11 * T4) *
              0.0356455407428531;
    // species 9: C2H5
    result += y[9] *
              (+3.24185905e+00 - 3.56905235e-03 * T + 4.82667202e-05 * T2 -
               5.85401009e-08 * T3 + 2.25804514e-11 * T4) *
              0.0344091941366733;
    // species 10: C2H6
    result += y[10] *
              (+3.29142572e+00 - 5.50154901e-03 * T + 5.99438458e-05 * T2 -
               7.08466469e-08 * T3 + 2.68685836e-11 * T4) *
              0.0332557366145660;
    // species 11: C6H6
    result += y[11] *
              (-4.95181368e-01 + 1.85020642e-02 * T + 7.38345881e-05 * T2 -
               1.18135741e-07 * T3 + 5.07210429e-11 * T4) *
              0.0128018024937911;
  } else {
    // species 1: H2
    result += y[1] *
              (+1.98711895e+00 + 7.36069465e-04 * T - 9.00982609e-08 * T2 -
               7.41224720e-13 * T3 + 6.58618037e-16 * T4) *
              0.4960317460317460;
    // species 3: CH3
    result += y[3] *
              (+1.97812060e+00 + 5.79785200e-03 * T - 1.97558000e-06 * T2 +
               3.07297900e-10 * T3 - 1.79174160e-14 * T4) *
              0.0665114732291320;
    // species 4: CH4
    result += y[4] *
              (+9.11786000e-01 + 9.60267960e-03 * T - 3.38387841e-06 * T2 +
               5.38797240e-10 * T3 - 3.19306807e-14 * T4) *
              0.0623324814560868;
    // species 5: C2H
    result += y[5] *
              (+2.66270248e+00 + 3.82492252e-03 * T - 1.36632500e-06 * T2 +
               2.13455040e-10 * T3 - 1.23216848e-14 * T4) *
              0.0399520575309628;
    // species 6: C2H2
    result += y[6] *
              (+3.65878489e+00 + 4.88396667e-03 * T - 1.60828888e-06 * T2 +
               2.46974544e-10 * T3 - 1.38605959e-14 * T4) *
              0.0384054074813734;
    // species 7: C2H3
    result += y[7] *
              (+3.15026763e+00 + 7.54021341e-03 * T - 2.62997847e-06 * T2 +
               4.15974048e-10 * T3 - 2.45407509e-14 * T4) *
              0.0369740442209569;
    // species 8: C2H4
    result += y[8] *
              (+2.99182724e+00 + 1.04833908e-02 * T - 3.71721342e-06 * T2 +
               5.94628366e-10 * T3 - 3.53630386e-14 * T4) *
              0.0356455407428531;
    // species 9: C2H5
    result += y[9] *
              (+3.32195633e+00 + 1.23930542e-02 * T - 4.39680960e-06 * T2 +
               7.03519917e-10 * T3 - 4.18435239e-14 * T4) *
              0.0344091941366733;
    // species 10: C2H6
    result += y[10] *
              (+3.04666411e+00 + 1.53538802e-02 * T - 5.47039485e-06 * T2 +
               8.77826544e-10 * T3 - 5.23167531e-14 * T4) *
              0.0332557366145660;
    // species 11: C6H6
    result += y[11] *
              (+1.00809576e+01 + 2.07176746e-02 * T - 7.52145991e-06 * T2 +
               1.22320984e-09 * T3 - 7.36091279e-14 * T4) *
              0.0128018024937911;
  }

  cvbs = result * 8.31446261815324e+07;
}

// Returns the mean enthalpy of the mixture in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBML(const amrex::Real T, const amrex::Real x[], amrex::Real& hbml)
{
  amrex::Real result = 0;
  amrex::Real hml[12];                       //  temporary storage
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // perform dot product
  for (int id = 0; id < 12; ++id) {
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
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 0: Ar
  result +=
    y[0] * (+2.50000000e+00 - 7.45375000e+02 * invT) * 0.0250312891113892;
  // species 2: H
  result +=
    y[2] * (+2.50000000e+00 + 2.54736600e+04 * invT) * 0.9920634920634921;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: H2
    result +=
      y[1] *
      (+3.50207268e+00 + 4.32737827e-05 * T - 8.78944480e-08 * T2 +
       8.43266552e-11 * T3 - 5.84719412e-15 * T4 - 1.04631279e+03 * invT) *
      0.4960317460317460;
    // species 3: CH3
    result +=
      y[3] *
      (+3.65717970e+00 + 1.06329895e-03 * T + 1.81946277e-06 * T2 -
       1.65452507e-09 * T3 + 4.93141480e-13 * T4 + 1.64227160e+04 * invT) *
      0.0665114732291320;
    // species 4: CH4
    result +=
      y[4] *
      (+5.14825732e+00 - 6.85012050e-03 * T + 1.64583138e-05 * T2 -
       1.22988085e-08 * T3 + 3.40194598e-12 * T4 - 1.02453222e+04 * invT) *
      0.0623324814560868;
    // species 5: C2H
    result +=
      y[5] *
      (+2.89867676e+00 + 6.64942445e-03 * T - 9.35777757e-06 * T2 +
       7.23711887e-09 * T3 - 2.15004702e-12 * T4 + 6.70616050e+04 * invT) *
      0.0399520575309628;
    // species 6: C2H2
    result +=
      y[6] *
      (+8.08679682e-01 + 1.16807881e-02 * T - 1.18390745e-05 * T2 +
       7.00382395e-09 * T3 - 1.70015033e-12 * T4 + 2.64289808e+04 * invT) *
      0.0384054074813734;
    // species 7: C2H3
    result +=
      y[7] *
      (+3.36377642e+00 + 1.32882861e-04 * T + 9.32069013e-06 * T2 -
       9.32467355e-09 * T3 + 3.03180352e-12 * T4 + 3.44749589e+04 * invT) *
      0.0369740442209569;
    // species 8: C2H4
    result +=
      y[8] *
      (+3.95920063e+00 - 3.78525686e-03 * T + 1.90329998e-05 * T2 -
       1.72897088e-08 * T3 + 5.39768380e-12 * T4 + 5.08977598e+03 * invT) *
      0.0356455407428531;
    // species 9: C2H5
    result +=
      y[9] *
      (+4.24185905e+00 - 1.78452617e-03 * T + 1.60889067e-05 * T2 -
       1.46350252e-08 * T3 + 4.51609028e-12 * T4 + 1.29690344e+04 * invT) *
      0.0344091941366733;
    // species 10: C2H6
    result +=
      y[10] *
      (+4.29142572e+00 - 2.75077450e-03 * T + 1.99812819e-05 * T2 -
       1.77116617e-08 * T3 + 5.37371672e-12 * T4 - 1.15222056e+04 * invT) *
      0.0332557366145660;
    // species 11: C6H6
    result +=
      y[11] *
      (+5.04818632e-01 + 9.25103210e-03 * T + 2.46115294e-05 * T2 -
       2.95339352e-08 * T3 + 1.01442086e-11 * T4 + 8.55247913e+03 * invT) *
      0.0128018024937911;
  } else {
    // species 1: H2
    result +=
      y[1] *
      (+2.98711895e+00 + 3.68034732e-04 * T - 3.00327536e-08 * T2 -
       1.85306180e-13 * T3 + 1.31723607e-16 * T4 - 8.35448659e+02 * invT) *
      0.4960317460317460;
    // species 3: CH3
    result +=
      y[3] *
      (+2.97812060e+00 + 2.89892600e-03 * T - 6.58526667e-07 * T2 +
       7.68244750e-11 * T3 - 3.58348320e-15 * T4 + 1.65095130e+04 * invT) *
      0.0665114732291320;
    // species 4: CH4
    result +=
      y[4] *
      (+1.91178600e+00 + 4.80133980e-03 * T - 1.12795947e-06 * T2 +
       1.34699310e-10 * T3 - 6.38613614e-15 * T4 - 1.00992136e+04 * invT) *
      0.0623324814560868;
    // species 5: C2H
    result +=
      y[5] *
      (+3.66270248e+00 + 1.91246126e-03 * T - 4.55441667e-07 * T2 +
       5.33637600e-11 * T3 - 2.46433696e-15 * T4 + 6.71683790e+04 * invT) *
      0.0399520575309628;
    // species 6: C2H2
    result +=
      y[6] *
      (+4.65878489e+00 + 2.44198333e-03 * T - 5.36096293e-07 * T2 +
       6.17436360e-11 * T3 - 2.77211918e-15 * T4 + 2.57594042e+04 * invT) *
      0.0384054074813734;
    // species 7: C2H3
    result +=
      y[7] *
      (+4.15026763e+00 + 3.77010670e-03 * T - 8.76659490e-07 * T2 +
       1.03993512e-10 * T3 - 4.90815018e-15 * T4 + 3.38566380e+04 * invT) *
      0.0369740442209569;
    // species 8: C2H4
    result +=
      y[8] *
      (+3.99182724e+00 + 5.24169540e-03 * T - 1.23907114e-06 * T2 +
       1.48657092e-10 * T3 - 7.07260772e-15 * T4 + 4.26865851e+03 * invT) *
      0.0356455407428531;
    // species 9: C2H5
    result +=
      y[9] *
      (+4.32195633e+00 + 6.19652710e-03 * T - 1.46560320e-06 * T2 +
       1.75879979e-10 * T3 - 8.36870478e-15 * T4 + 1.21759475e+04 * invT) *
      0.0344091941366733;
    // species 10: C2H6
    result +=
      y[10] *
      (+4.04666411e+00 + 7.67694010e-03 * T - 1.82346495e-06 * T2 +
       2.19456636e-10 * T3 - 1.04633506e-14 * T4 - 1.24473499e+04 * invT) *
      0.0332557366145660;
    // species 11: C6H6
    result +=
      y[11] *
      (+1.10809576e+01 + 1.03588373e-02 * T - 2.50715330e-06 * T2 +
       3.05802460e-10 * T3 - 1.47218256e-14 * T4 + 4.30641035e+03 * invT) *
      0.0128018024937911;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  hbms = result * RT;
}

// get mean internal energy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBML(const amrex::Real T, const amrex::Real x[], amrex::Real& ubml)
{
  amrex::Real result = 0;
  amrex::Real uml[12];                       //  temporary energy array
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // perform dot product
  for (int id = 0; id < 12; ++id) {
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
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 0: Ar
  result +=
    y[0] * (+1.50000000e+00 - 7.45375000e+02 * invT) * 0.0250312891113892;
  // species 2: H
  result +=
    y[2] * (+1.50000000e+00 + 2.54736600e+04 * invT) * 0.9920634920634921;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: H2
    result +=
      y[1] *
      (+2.50207268e+00 + 4.32737827e-05 * T - 8.78944480e-08 * T2 +
       8.43266552e-11 * T3 - 5.84719412e-15 * T4 - 1.04631279e+03 * invT) *
      0.4960317460317460;
    // species 3: CH3
    result +=
      y[3] *
      (+2.65717970e+00 + 1.06329895e-03 * T + 1.81946277e-06 * T2 -
       1.65452507e-09 * T3 + 4.93141480e-13 * T4 + 1.64227160e+04 * invT) *
      0.0665114732291320;
    // species 4: CH4
    result +=
      y[4] *
      (+4.14825732e+00 - 6.85012050e-03 * T + 1.64583138e-05 * T2 -
       1.22988085e-08 * T3 + 3.40194598e-12 * T4 - 1.02453222e+04 * invT) *
      0.0623324814560868;
    // species 5: C2H
    result +=
      y[5] *
      (+1.89867676e+00 + 6.64942445e-03 * T - 9.35777757e-06 * T2 +
       7.23711887e-09 * T3 - 2.15004702e-12 * T4 + 6.70616050e+04 * invT) *
      0.0399520575309628;
    // species 6: C2H2
    result +=
      y[6] *
      (-1.91320318e-01 + 1.16807881e-02 * T - 1.18390745e-05 * T2 +
       7.00382395e-09 * T3 - 1.70015033e-12 * T4 + 2.64289808e+04 * invT) *
      0.0384054074813734;
    // species 7: C2H3
    result +=
      y[7] *
      (+2.36377642e+00 + 1.32882861e-04 * T + 9.32069013e-06 * T2 -
       9.32467355e-09 * T3 + 3.03180352e-12 * T4 + 3.44749589e+04 * invT) *
      0.0369740442209569;
    // species 8: C2H4
    result +=
      y[8] *
      (+2.95920063e+00 - 3.78525686e-03 * T + 1.90329998e-05 * T2 -
       1.72897088e-08 * T3 + 5.39768380e-12 * T4 + 5.08977598e+03 * invT) *
      0.0356455407428531;
    // species 9: C2H5
    result +=
      y[9] *
      (+3.24185905e+00 - 1.78452617e-03 * T + 1.60889067e-05 * T2 -
       1.46350252e-08 * T3 + 4.51609028e-12 * T4 + 1.29690344e+04 * invT) *
      0.0344091941366733;
    // species 10: C2H6
    result +=
      y[10] *
      (+3.29142572e+00 - 2.75077450e-03 * T + 1.99812819e-05 * T2 -
       1.77116617e-08 * T3 + 5.37371672e-12 * T4 - 1.15222056e+04 * invT) *
      0.0332557366145660;
    // species 11: C6H6
    result +=
      y[11] *
      (-4.95181368e-01 + 9.25103210e-03 * T + 2.46115294e-05 * T2 -
       2.95339352e-08 * T3 + 1.01442086e-11 * T4 + 8.55247913e+03 * invT) *
      0.0128018024937911;
  } else {
    // species 1: H2
    result +=
      y[1] *
      (+1.98711895e+00 + 3.68034732e-04 * T - 3.00327536e-08 * T2 -
       1.85306180e-13 * T3 + 1.31723607e-16 * T4 - 8.35448659e+02 * invT) *
      0.4960317460317460;
    // species 3: CH3
    result +=
      y[3] *
      (+1.97812060e+00 + 2.89892600e-03 * T - 6.58526667e-07 * T2 +
       7.68244750e-11 * T3 - 3.58348320e-15 * T4 + 1.65095130e+04 * invT) *
      0.0665114732291320;
    // species 4: CH4
    result +=
      y[4] *
      (+9.11786000e-01 + 4.80133980e-03 * T - 1.12795947e-06 * T2 +
       1.34699310e-10 * T3 - 6.38613614e-15 * T4 - 1.00992136e+04 * invT) *
      0.0623324814560868;
    // species 5: C2H
    result +=
      y[5] *
      (+2.66270248e+00 + 1.91246126e-03 * T - 4.55441667e-07 * T2 +
       5.33637600e-11 * T3 - 2.46433696e-15 * T4 + 6.71683790e+04 * invT) *
      0.0399520575309628;
    // species 6: C2H2
    result +=
      y[6] *
      (+3.65878489e+00 + 2.44198333e-03 * T - 5.36096293e-07 * T2 +
       6.17436360e-11 * T3 - 2.77211918e-15 * T4 + 2.57594042e+04 * invT) *
      0.0384054074813734;
    // species 7: C2H3
    result +=
      y[7] *
      (+3.15026763e+00 + 3.77010670e-03 * T - 8.76659490e-07 * T2 +
       1.03993512e-10 * T3 - 4.90815018e-15 * T4 + 3.38566380e+04 * invT) *
      0.0369740442209569;
    // species 8: C2H4
    result +=
      y[8] *
      (+2.99182724e+00 + 5.24169540e-03 * T - 1.23907114e-06 * T2 +
       1.48657092e-10 * T3 - 7.07260772e-15 * T4 + 4.26865851e+03 * invT) *
      0.0356455407428531;
    // species 9: C2H5
    result +=
      y[9] *
      (+3.32195633e+00 + 6.19652710e-03 * T - 1.46560320e-06 * T2 +
       1.75879979e-10 * T3 - 8.36870478e-15 * T4 + 1.21759475e+04 * invT) *
      0.0344091941366733;
    // species 10: C2H6
    result +=
      y[10] *
      (+3.04666411e+00 + 7.67694010e-03 * T - 1.82346495e-06 * T2 +
       2.19456636e-10 * T3 - 1.04633506e-14 * T4 - 1.24473499e+04 * invT) *
      0.0332557366145660;
    // species 11: C6H6
    result +=
      y[11] *
      (+1.00809576e+01 + 1.03588373e-02 * T - 2.50715330e-06 * T2 +
       3.05802460e-10 * T3 - 1.47218256e-14 * T4 + 4.30641035e+03 * invT) *
      0.0128018024937911;
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
  amrex::Real sor[12]; //  temporary storage
  speciesEntropy(sor, T);

  // Compute Eq 42
  for (int id = 0; id < 12; ++id) {
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
  amrex::Real sor[12]; //  temporary storage
  amrex::Real x[12];   //  need a ytx conversion
  amrex::Real YOW = 0; // See Eq 4, 6 in CK Manual

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 12; i++) {
    YOW += y[i] * imw(i);
  }
  // Now compute y to x conversion
  x[0] = y[0] / (3.9950000000000003e+01 * YOW);
  x[1] = y[1] / (2.0160000000000000e+00 * YOW);
  x[2] = y[2] / (1.0080000000000000e+00 * YOW);
  x[3] = y[3] / (1.5035000000000000e+01 * YOW);
  x[4] = y[4] / (1.6042999999999999e+01 * YOW);
  x[5] = y[5] / (2.5029999999999998e+01 * YOW);
  x[6] = y[6] / (2.6037999999999997e+01 * YOW);
  x[7] = y[7] / (2.7045999999999999e+01 * YOW);
  x[8] = y[8] / (2.8053999999999998e+01 * YOW);
  x[9] = y[9] / (2.9061999999999998e+01 * YOW);
  x[10] = y[10] / (3.0070000000000000e+01 * YOW);
  x[11] = y[11] / (7.8114000000000004e+01 * YOW);
  speciesEntropy(sor, T);
  // Perform computation in Eq 42 and 43
  for (int i = 0; i < 12; i++) {
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
  amrex::Real tmin = 200.0;  // max lower bound for thermo def
  amrex::Real tmax = 6000.0; // min upper bound for thermo def
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
  amrex::Real tmin = 200.0;  // max lower bound for thermo def
  amrex::Real tmax = 6000.0; // min upper bound for thermo def
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
  XW += x[0] * 3.9950000000000003e+01;     // Ar
  XW += x[1] * 2.0160000000000000e+00;     // H2
  XW += x[2] * 1.0080000000000000e+00;     // H
  XW += x[3] * 1.5035000000000000e+01;     // CH3
  XW += x[4] * 1.6042999999999999e+01;     // CH4
  XW += x[5] * 2.5029999999999998e+01;     // C2H
  XW += x[6] * 2.6037999999999997e+01;     // C2H2
  XW += x[7] * 2.7045999999999999e+01;     // C2H3
  XW += x[8] * 2.8053999999999998e+01;     // C2H4
  XW += x[9] * 2.9061999999999998e+01;     // C2H5
  XW += x[10] * 3.0070000000000000e+01;    // C2H6
  XW += x[11] * 7.8114000000000004e+01;    // C6H6
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

  for (int i = 0; i < 12; i++) {
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
  W += c[0] * 3.9950000000000003e+01;  // Ar
  W += c[1] * 2.0160000000000000e+00;  // H2
  W += c[2] * 1.0080000000000000e+00;  // H
  W += c[3] * 1.5035000000000000e+01;  // CH3
  W += c[4] * 1.6042999999999999e+01;  // CH4
  W += c[5] * 2.5029999999999998e+01;  // C2H
  W += c[6] * 2.6037999999999997e+01;  // C2H2
  W += c[7] * 2.7045999999999999e+01;  // C2H3
  W += c[8] * 2.8053999999999998e+01;  // C2H4
  W += c[9] * 2.9061999999999998e+01;  // C2H5
  W += c[10] * 3.0070000000000000e+01; // C2H6
  W += c[11] * 7.8114000000000004e+01; // C6H6

  for (int id = 0; id < 12; ++id) {
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
  XW += x[0] * 3.9950000000000003e+01;       // Ar
  XW += x[1] * 2.0160000000000000e+00;       // H2
  XW += x[2] * 1.0080000000000000e+00;       // H
  XW += x[3] * 1.5035000000000000e+01;       // CH3
  XW += x[4] * 1.6042999999999999e+01;       // CH4
  XW += x[5] * 2.5029999999999998e+01;       // C2H
  XW += x[6] * 2.6037999999999997e+01;       // C2H2
  XW += x[7] * 2.7045999999999999e+01;       // C2H3
  XW += x[8] * 2.8053999999999998e+01;       // C2H4
  XW += x[9] * 2.9061999999999998e+01;       // C2H5
  XW += x[10] * 3.0070000000000000e+01;      // C2H6
  XW += x[11] * 7.8114000000000004e+01;      // C6H6
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

  for (int i = 0; i < 12; i++) {
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
  W += c[0] * 3.9950000000000003e+01;  // Ar
  W += c[1] * 2.0160000000000000e+00;  // H2
  W += c[2] * 1.0080000000000000e+00;  // H
  W += c[3] * 1.5035000000000000e+01;  // CH3
  W += c[4] * 1.6042999999999999e+01;  // CH4
  W += c[5] * 2.5029999999999998e+01;  // C2H
  W += c[6] * 2.6037999999999997e+01;  // C2H2
  W += c[7] * 2.7045999999999999e+01;  // C2H3
  W += c[8] * 2.8053999999999998e+01;  // C2H4
  W += c[9] * 2.9061999999999998e+01;  // C2H5
  W += c[10] * 3.0070000000000000e+01; // C2H6
  W += c[11] * 7.8114000000000004e+01; // C6H6

  for (int id = 0; id < 12; ++id) {
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

  for (int i = 0; i < 12; i++) {
    YOW += y[i] * imw(i);
  }

  wtm = 1.0 / YOW;
}

// given x[species]: mole fractions
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWX(const amrex::Real x[], amrex::Real& wtm)
{
  amrex::Real XW = 0;                   //  see Eq 4 in CK Manual
  XW += x[0] * 3.9950000000000003e+01;  // Ar
  XW += x[1] * 2.0160000000000000e+00;  // H2
  XW += x[2] * 1.0080000000000000e+00;  // H
  XW += x[3] * 1.5035000000000000e+01;  // CH3
  XW += x[4] * 1.6042999999999999e+01;  // CH4
  XW += x[5] * 2.5029999999999998e+01;  // C2H
  XW += x[6] * 2.6037999999999997e+01;  // C2H2
  XW += x[7] * 2.7045999999999999e+01;  // C2H3
  XW += x[8] * 2.8053999999999998e+01;  // C2H4
  XW += x[9] * 2.9061999999999998e+01;  // C2H5
  XW += x[10] * 3.0070000000000000e+01; // C2H6
  XW += x[11] * 7.8114000000000004e+01; // C6H6
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
  W += c[0] * 3.9950000000000003e+01;  // Ar
  W += c[1] * 2.0160000000000000e+00;  // H2
  W += c[2] * 1.0080000000000000e+00;  // H
  W += c[3] * 1.5035000000000000e+01;  // CH3
  W += c[4] * 1.6042999999999999e+01;  // CH4
  W += c[5] * 2.5029999999999998e+01;  // C2H
  W += c[6] * 2.6037999999999997e+01;  // C2H2
  W += c[7] * 2.7045999999999999e+01;  // C2H3
  W += c[8] * 2.8053999999999998e+01;  // C2H4
  W += c[9] * 2.9061999999999998e+01;  // C2H5
  W += c[10] * 3.0070000000000000e+01; // C2H6
  W += c[11] * 7.8114000000000004e+01; // C6H6

  for (int id = 0; id < 12; ++id) {
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

  for (int i = 0; i < 12; i++) {
    YOW += y[i] * imw(i);
  }

  amrex::Real YOWINV = 1.0 / YOW;

  for (int i = 0; i < 12; i++) {
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
  for (int i = 0; i < 12; i++) {
    c[i] = y[i] * imw(i);
  }
  for (int i = 0; i < 12; i++) {
    YOW += c[i];
  }

  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // Now compute conversion

  for (int i = 0; i < 12; i++) {
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

  for (int i = 0; i < 12; i++) {
    c[i] = rho * y[i] * imw(i);
  }
}

// convert x[species] (mole fracs) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTY(const amrex::Real x[], amrex::Real y[])
{
  amrex::Real XW = 0; // See Eq 4, 9 in CK Manual
  // Compute mean molecular wt first
  XW += x[0] * 3.9950000000000003e+01;  // Ar
  XW += x[1] * 2.0160000000000000e+00;  // H2
  XW += x[2] * 1.0080000000000000e+00;  // H
  XW += x[3] * 1.5035000000000000e+01;  // CH3
  XW += x[4] * 1.6042999999999999e+01;  // CH4
  XW += x[5] * 2.5029999999999998e+01;  // C2H
  XW += x[6] * 2.6037999999999997e+01;  // C2H2
  XW += x[7] * 2.7045999999999999e+01;  // C2H3
  XW += x[8] * 2.8053999999999998e+01;  // C2H4
  XW += x[9] * 2.9061999999999998e+01;  // C2H5
  XW += x[10] * 3.0070000000000000e+01; // C2H6
  XW += x[11] * 7.8114000000000004e+01; // C6H6
  // Now compute conversion
  amrex::Real XWinv = 1.0 / XW;
  y[0] = x[0] * 3.9950000000000003e+01 * XWinv;
  y[1] = x[1] * 2.0160000000000000e+00 * XWinv;
  y[2] = x[2] * 1.0080000000000000e+00 * XWinv;
  y[3] = x[3] * 1.5035000000000000e+01 * XWinv;
  y[4] = x[4] * 1.6042999999999999e+01 * XWinv;
  y[5] = x[5] * 2.5029999999999998e+01 * XWinv;
  y[6] = x[6] * 2.6037999999999997e+01 * XWinv;
  y[7] = x[7] * 2.7045999999999999e+01 * XWinv;
  y[8] = x[8] * 2.8053999999999998e+01 * XWinv;
  y[9] = x[9] * 2.9061999999999998e+01 * XWinv;
  y[10] = x[10] * 3.0070000000000000e+01 * XWinv;
  y[11] = x[11] * 7.8114000000000004e+01 * XWinv;
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
  for (int id = 0; id < 12; ++id) {
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
  XW += x[0] * 3.9950000000000003e+01;  // Ar
  XW += x[1] * 2.0160000000000000e+00;  // H2
  XW += x[2] * 1.0080000000000000e+00;  // H
  XW += x[3] * 1.5035000000000000e+01;  // CH3
  XW += x[4] * 1.6042999999999999e+01;  // CH4
  XW += x[5] * 2.5029999999999998e+01;  // C2H
  XW += x[6] * 2.6037999999999997e+01;  // C2H2
  XW += x[7] * 2.7045999999999999e+01;  // C2H3
  XW += x[8] * 2.8053999999999998e+01;  // C2H4
  XW += x[9] * 2.9061999999999998e+01;  // C2H5
  XW += x[10] * 3.0070000000000000e+01; // C2H6
  XW += x[11] * 7.8114000000000004e+01; // C6H6
  ROW = rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 12; ++id) {
    c[id] = x[id] * ROW;
  }
}

// convert c[species] (molar conc) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTX(const amrex::Real c[], amrex::Real x[])
{
  amrex::Real sumC = 0;

  // compute sum of c
  for (int id = 0; id < 12; ++id) {
    sumC += c[id];
  }

  //  See Eq 13
  amrex::Real sumCinv = 1.0 / sumC;
  for (int id = 0; id < 12; ++id) {
    x[id] = c[id] * sumCinv;
  }
}

// convert c[species] (molar conc) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTY(const amrex::Real c[], amrex::Real y[])
{
  amrex::Real CW = 0; // See Eq 12 in CK Manual
  // compute denominator in eq 12 first
  CW += c[0] * 3.9950000000000003e+01;  // Ar
  CW += c[1] * 2.0160000000000000e+00;  // H2
  CW += c[2] * 1.0080000000000000e+00;  // H
  CW += c[3] * 1.5035000000000000e+01;  // CH3
  CW += c[4] * 1.6042999999999999e+01;  // CH4
  CW += c[5] * 2.5029999999999998e+01;  // C2H
  CW += c[6] * 2.6037999999999997e+01;  // C2H2
  CW += c[7] * 2.7045999999999999e+01;  // C2H3
  CW += c[8] * 2.8053999999999998e+01;  // C2H4
  CW += c[9] * 2.9061999999999998e+01;  // C2H5
  CW += c[10] * 3.0070000000000000e+01; // C2H6
  CW += c[11] * 7.8114000000000004e+01; // C6H6
  // Now compute conversion
  amrex::Real CWinv = 1.0 / CW;
  y[0] = c[0] * 3.9950000000000003e+01 * CWinv;
  y[1] = c[1] * 2.0160000000000000e+00 * CWinv;
  y[2] = c[2] * 1.0080000000000000e+00 * CWinv;
  y[3] = c[3] * 1.5035000000000000e+01 * CWinv;
  y[4] = c[4] * 1.6042999999999999e+01 * CWinv;
  y[5] = c[5] * 2.5029999999999998e+01 * CWinv;
  y[6] = c[6] * 2.6037999999999997e+01 * CWinv;
  y[7] = c[7] * 2.7045999999999999e+01 * CWinv;
  y[8] = c[8] * 2.8053999999999998e+01 * CWinv;
  y[9] = c[9] * 2.9061999999999998e+01 * CWinv;
  y[10] = c[10] * 3.0070000000000000e+01 * CWinv;
  y[11] = c[11] * 7.8114000000000004e+01 * CWinv;
}

// get specific heat at constant volume as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVML(const amrex::Real T, amrex::Real cvml[])
{
  cv_R(cvml, T);

  // convert to chemkin units
  for (int id = 0; id < 12; ++id) {
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
  for (int id = 0; id < 12; ++id) {
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
  for (int id = 0; id < 12; ++id) {
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
  for (int id = 0; id < 12; ++id) {
    hml[id] *= RT;
  }
}

// Returns the standard-state entropies in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSML(const amrex::Real T, amrex::Real sml[])
{
  speciesEntropy(sml, T);

  // convert to chemkin units
  for (int id = 0; id < 12; ++id) {
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
  cvms[0] *= 2.081217176008320e+06;  // Ar
  cvms[1] *= 4.124237409798234e+07;  // H2
  cvms[2] *= 8.248474819596468e+07;  // H
  cvms[3] *= 5.530071578419182e+06;  // CH3
  cvms[4] *= 5.182610869633635e+06;  // CH4
  cvms[5] *= 3.321798888594982e+06;  // C2H
  cvms[6] *= 3.193203248388218e+06;  // C2H2
  cvms[7] *= 3.074193085170909e+06;  // C2H3
  cvms[8] *= 2.963735160103101e+06;  // C2H4
  cvms[9] *= 2.860939583701480e+06;  // C2H5
  cvms[10] *= 2.765035789209591e+06; // C2H6
  cvms[11] *= 1.064401082796072e+06; // C6H6
}

// Returns the specific heats at constant pressure
// in mass units (Eq. 26)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPMS(const amrex::Real T, amrex::Real cpms[])
{
  cp_R(cpms, T);
  // multiply by R/molecularweight
  cpms[0] *= 2.081217176008320e+06;  // Ar
  cpms[1] *= 4.124237409798234e+07;  // H2
  cpms[2] *= 8.248474819596468e+07;  // H
  cpms[3] *= 5.530071578419182e+06;  // CH3
  cpms[4] *= 5.182610869633635e+06;  // CH4
  cpms[5] *= 3.321798888594982e+06;  // C2H
  cpms[6] *= 3.193203248388218e+06;  // C2H2
  cpms[7] *= 3.074193085170909e+06;  // C2H3
  cpms[8] *= 2.963735160103101e+06;  // C2H4
  cpms[9] *= 2.860939583701480e+06;  // C2H5
  cpms[10] *= 2.765035789209591e+06; // C2H6
  cpms[11] *= 1.064401082796072e+06; // C6H6
}

// Returns internal energy in mass units (Eq 30.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUMS(const amrex::Real T, amrex::Real ums[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesInternalEnergy(ums, T);

  for (int i = 0; i < 12; i++) {
    ums[i] *= RT * imw(i);
  }
}

// Returns enthalpy in mass units (Eq 27.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHMS(const amrex::Real T, amrex::Real hms[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesEnthalpy(hms, T);

  for (int i = 0; i < 12; i++) {
    hms[i] *= RT * imw(i);
  }
}

// Returns the entropies in mass units (Eq 28.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSMS(const amrex::Real T, amrex::Real sms[])
{
  speciesEntropy(sms, T);
  // multiply by R/molecularweight
  sms[0] *= 2.081217176008320e+06;  // Ar
  sms[1] *= 4.124237409798234e+07;  // H2
  sms[2] *= 8.248474819596468e+07;  // H
  sms[3] *= 5.530071578419182e+06;  // CH3
  sms[4] *= 5.182610869633635e+06;  // CH4
  sms[5] *= 3.321798888594982e+06;  // C2H
  sms[6] *= 3.193203248388218e+06;  // C2H2
  sms[7] *= 3.074193085170909e+06;  // C2H3
  sms[8] *= 2.963735160103101e+06;  // C2H4
  sms[9] *= 2.860939583701480e+06;  // C2H5
  sms[10] *= 2.765035789209591e+06; // C2H6
  sms[11] *= 1.064401082796072e+06; // C6H6
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

  // reaction 0: CH3 + H <=> CH4
  qf[0] = sc[2] * sc[3];
  qr[0] = sc[4];

  // reaction 1: CH4 + H <=> CH3 + H2
  qf[1] = sc[2] * sc[4];
  qr[1] = sc[1] * sc[3];

  // reaction 2: 2 CH3 <=> C2H6
  qf[2] = (sc[3] * sc[3]);
  qr[2] = sc[10];

  // reaction 3: 2 CH3 <=> C2H5 + H
  qf[3] = (sc[3] * sc[3]);
  qr[3] = sc[2] * sc[9];

  // reaction 4: C2H6 + H <=> C2H5 + H2
  qf[4] = sc[2] * sc[10];
  qr[4] = sc[1] * sc[9];

  // reaction 5: C2H4 + H <=> C2H5
  qf[5] = sc[2] * sc[8];
  qr[5] = sc[9];

  // reaction 6: C2H4 + H <=> C2H3 + H2
  qf[6] = sc[2] * sc[8];
  qr[6] = sc[1] * sc[7];

  // reaction 7: C2H2 + H <=> C2H + H2
  qf[7] = sc[2] * sc[6];
  qr[7] = sc[1] * sc[5];

  // reaction 8: C2H2 + H <=> C2H3
  qf[8] = sc[2] * sc[6];
  qr[8] = sc[7];

  // reaction 9: 3 C2H2 <=> C6H6
  qf[9] = sc[6] * sc[6] * sc[6];
  qr[9] = sc[11];

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 12; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[12];
  gibbs(g_RT, T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 * invT;
  amrex::Real refCinv = 1 / refC;

  // Evaluate the kfs
  amrex::Real k_f, Corr;

  // reaction 0:  CH3 + H <=> CH4
  k_f = 3.674e+17 * exp((-2.713) * logT - (23.0079704035475) * invT);
  qf[0] *= k_f;
  qr[0] *= k_f * exp(-(g_RT[2] + g_RT[3] - g_RT[4])) * (refC);
  // reaction 1:  CH4 + H <=> CH3 + H2
  k_f = 943.9 * exp((1.409) * logT - (3977.3841152395) * invT);
  qf[1] *= k_f;
  qr[1] *= k_f * exp(-(-g_RT[1] + g_RT[2] - g_RT[3] + g_RT[4]));
  // reaction 2:  2 CH3 <=> C2H6
  k_f = 6028000000000 * exp((0.1215) * logT - (2274.33727303233) * invT);
  qf[2] *= k_f;
  qr[2] *= k_f * exp(-(2.000000 * g_RT[3] - g_RT[10])) * (refC);
  // reaction 3:  2 CH3 <=> C2H5 + H
  k_f = 875200000 * exp((-0.9293) * logT - (52.3903392984072) * invT);
  qf[3] *= k_f;
  qr[3] *= k_f * exp(-(-g_RT[2] + 2.000000 * g_RT[3] - g_RT[9]));
  // reaction 4:  C2H6 + H <=> C2H5 + H2
  k_f = 126.5 * exp((2.281) * logT - (5787.47274343287) * invT);
  qf[4] *= k_f;
  qr[4] *= k_f * exp(-(-g_RT[1] + g_RT[2] - g_RT[9] + g_RT[10]));
  // reaction 5:  C2H4 + H <=> C2H5
  k_f = 731500000 * exp((-0.6492) * logT - (13.2298836612139) * invT);
  qf[5] *= k_f;
  qr[5] *= k_f * exp(-(g_RT[2] + g_RT[8] - g_RT[9])) * (refC);
  // reaction 6:  C2H4 + H <=> C2H3 + H2
  k_f = 24.64 * exp((1.851) * logT - (1875.03532980296) * invT);
  qf[6] *= k_f;
  qr[6] *= k_f * exp(-(-g_RT[1] + g_RT[2] - g_RT[7] + g_RT[8]));
  // reaction 7:  C2H2 + H <=> C2H + H2
  k_f = 3207000000 * exp((-0.0941) * logT - (29562.7763993308) * invT);
  qf[7] *= k_f;
  qr[7] *= k_f * exp(-(-g_RT[1] + g_RT[2] - g_RT[5] + g_RT[6]));
  // reaction 8:  C2H2 + H <=> C2H3
  k_f = 86330000 * exp((-0.2163) * logT - (1321.78564942492) * invT);
  qf[8] *= k_f;
  qr[8] *= k_f * exp(-(g_RT[2] + g_RT[6] - g_RT[7])) * (refC);
  // reaction 9:  3 C2H2 <=> C6H6
  k_f = 0.07581 * exp((-0.001516) * logT - (2.90335810528822) * invT);
  qf[9] *= k_f;
  qr[9] *= k_f * exp(-(3.000000 * g_RT[6] - g_RT[11]));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
productionRate(amrex::Real* wdot, const amrex::Real* sc, const amrex::Real T)
{
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  const amrex::Real refC = 101325 / 8.31446 * invT;
  const amrex::Real refCinv = 1 / refC;

  for (int i = 0; i < 12; ++i) {
    wdot[i] = 0.0;
  }

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 12; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[12];
  gibbs(g_RT, T);

  {
    // reaction 0:  CH3 + H <=> CH4
    const amrex::Real k_f =
      3.674e+17 * exp((-2.713) * logT - (23.0079704035475) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[3]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] + g_RT[3] - g_RT[4])) * (refC) * (sc[4]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;
  }

  {
    // reaction 1:  CH4 + H <=> CH3 + H2
    const amrex::Real k_f =
      943.9 * exp((1.409) * logT - (3977.3841152395) * invT);
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
    // reaction 2:  2 CH3 <=> C2H6
    const amrex::Real k_f =
      6028000000000 * exp((0.1215) * logT - (2274.33727303233) * invT);
    const amrex::Real qf = k_f * ((sc[3] * sc[3]));
    const amrex::Real qr =
      k_f * exp(-(2.000000 * g_RT[3] - g_RT[10])) * (refC) * (sc[10]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= 2.000000 * qdot;
    wdot[10] += qdot;
  }

  {
    // reaction 3:  2 CH3 <=> C2H5 + H
    const amrex::Real k_f =
      875200000 * exp((-0.9293) * logT - (52.3903392984072) * invT);
    const amrex::Real qf = k_f * ((sc[3] * sc[3]));
    const amrex::Real qr =
      k_f * exp(-(-g_RT[2] + 2.000000 * g_RT[3] - g_RT[9])) * (sc[2] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[3] -= 2.000000 * qdot;
    wdot[9] += qdot;
  }

  {
    // reaction 4:  C2H6 + H <=> C2H5 + H2
    const amrex::Real k_f =
      126.5 * exp((2.281) * logT - (5787.47274343287) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[10]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[2] - g_RT[9] + g_RT[10])) * (sc[1] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[9] += qdot;
    wdot[10] -= qdot;
  }

  {
    // reaction 5:  C2H4 + H <=> C2H5
    const amrex::Real k_f =
      731500000 * exp((-0.6492) * logT - (13.2298836612139) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[8]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] + g_RT[8] - g_RT[9])) * (refC) * (sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[8] -= qdot;
    wdot[9] += qdot;
  }

  {
    // reaction 6:  C2H4 + H <=> C2H3 + H2
    const amrex::Real k_f =
      24.64 * exp((1.851) * logT - (1875.03532980296) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[8]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[2] - g_RT[7] + g_RT[8])) * (sc[1] * sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[7] += qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 7:  C2H2 + H <=> C2H + H2
    const amrex::Real k_f =
      3207000000 * exp((-0.0941) * logT - (29562.7763993308) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[6]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[2] - g_RT[5] + g_RT[6])) * (sc[1] * sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[5] += qdot;
    wdot[6] -= qdot;
  }

  {
    // reaction 8:  C2H2 + H <=> C2H3
    const amrex::Real k_f =
      86330000 * exp((-0.2163) * logT - (1321.78564942492) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[6]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] + g_RT[6] - g_RT[7])) * (refC) * (sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[6] -= qdot;
    wdot[7] += qdot;
  }

  {
    // reaction 9:  3 C2H2 <=> C6H6
    const amrex::Real k_f =
      0.07581 * exp((-0.001516) * logT - (2.90335810528822) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[6] * sc[6]);
    const amrex::Real qr =
      k_f * exp(-(3.000000 * g_RT[6] - g_RT[11])) * (sc[11]);
    const amrex::Real qdot = qf - qr;
    wdot[6] -= 3.000000 * qdot;
    wdot[11] += qdot;
  }
}

// compute the production rate for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWC(const amrex::Real T, amrex::Real C[], amrex::Real wdot[])
{

  // convert to SI
  for (int id = 0; id < 12; ++id) {
    C[id] *= 1.0e6;
  }

  // convert to chemkin units
  productionRate(wdot, C, T);

  // convert to chemkin units
  for (int id = 0; id < 12; ++id) {
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
  amrex::Real c[12]; // temporary storage
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 12; i++) {
    YOW += y[i] * imw(i);
  }
  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // multiply by 1e6 so c goes to SI
  PWORT *= 1e6;
  // Now compute conversion (and go to SI)
  for (int i = 0; i < 12; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 12; ++id) {
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
  amrex::Real c[12]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 12; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 12; ++id) {
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
  amrex::Real c[12]; // temporary storage

  // See Eq 8 with an extra 1e6 so c goes to SI
  for (int i = 0; i < 12; i++) {
    c[i] = 1e6 * rho * y[i] * imw(i);
  }

  // call productionRate
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 12; ++id) {
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
  amrex::Real c[12];  // temporary storage
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 3.9950000000000003e+01;  // Ar
  XW += x[1] * 2.0160000000000000e+00;  // H2
  XW += x[2] * 1.0080000000000000e+00;  // H
  XW += x[3] * 1.5035000000000000e+01;  // CH3
  XW += x[4] * 1.6042999999999999e+01;  // CH4
  XW += x[5] * 2.5029999999999998e+01;  // C2H
  XW += x[6] * 2.6037999999999997e+01;  // C2H2
  XW += x[7] * 2.7045999999999999e+01;  // C2H3
  XW += x[8] * 2.8053999999999998e+01;  // C2H4
  XW += x[9] * 2.9061999999999998e+01;  // C2H5
  XW += x[10] * 3.0070000000000000e+01; // C2H6
  XW += x[11] * 7.8114000000000004e+01; // C6H6
  // Extra 1e6 factor to take c to SI
  ROW = 1e6 * rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 12; ++id) {
    c[id] = x[id] * ROW;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 12; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRG(int kcharge[])
{
  kcharge[0] = 0;  // Ar
  kcharge[1] = 0;  // H2
  kcharge[2] = 0;  // H
  kcharge[3] = 0;  // CH3
  kcharge[4] = 0;  // CH4
  kcharge[5] = 0;  // C2H
  kcharge[6] = 0;  // C2H2
  kcharge[7] = 0;  // C2H3
  kcharge[8] = 0;  // C2H4
  kcharge[9] = 0;  // C2H5
  kcharge[10] = 0; // C2H6
  kcharge[11] = 0; // C6H6
}

//  species charge per unit mass
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRGMASS(amrex::Real zk[])
{

  int kchrg[12];
  CKCHRG(kchrg);

  for (int id = 0; id < 12; ++id) {
    zk[id] = 6.02214076e+23 * 1.60217663e-19 * kchrg[id] * imw(id);
  }
}

// compute an approx to the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian_precond(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int HP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 169; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[12];
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
  for (int k = 0; k < 12; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[12];
  gibbs(g_RT, T);

  // compute the species enthalpy
  amrex::Real h_RT[12];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[12];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 0: CH3 + H <=> CH4
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[3];
  k_f = 3.674e+17 * exp(-2.713 * logT - (23.0079704035475) * invT);
  dlnkfdT = -2.713 * invT + (23.0079704035475) * invT2;
  // reverse
  phi_r = sc[4];
  Kc = refCinv * exp(g_RT[2] + g_RT[3] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[3]) + (h_RT[4]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[3] -= q; // CH3
  wdot[4] += q; // CH4
  // d()/d[H]
  dqdci = +k_f * sc[3];
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[29] -= dqdci; // dwdot[CH3]/d[H]
  J[30] += dqdci; // dwdot[CH4]/d[H]
  // d()/d[CH3]
  dqdci = +k_f * sc[2];
  J[41] -= dqdci; // dwdot[H]/d[CH3]
  J[42] -= dqdci; // dwdot[CH3]/d[CH3]
  J[43] += dqdci; // dwdot[CH4]/d[CH3]
  // d()/d[CH4]
  dqdci = -k_r;
  J[54] -= dqdci; // dwdot[H]/d[CH4]
  J[55] -= dqdci; // dwdot[CH3]/d[CH4]
  J[56] += dqdci; // dwdot[CH4]/d[CH4]
  // d()/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[159] -= dqdT; // dwdot[CH3]/dT
  J[160] += dqdT; // dwdot[CH4]/dT

  // reaction 1: CH4 + H <=> CH3 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[4];
  k_f = 943.9 * exp(1.409 * logT - (3977.3841152395) * invT);
  dlnkfdT = 1.409 * invT + (3977.3841152395) * invT2;
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
  wdot[1] += q; // H2
  wdot[2] -= q; // H
  wdot[3] += q; // CH3
  wdot[4] -= q; // CH4
  // d()/d[H2]
  dqdci = -k_r * sc[3];
  J[14] += dqdci; // dwdot[H2]/d[H2]
  J[15] -= dqdci; // dwdot[H]/d[H2]
  J[16] += dqdci; // dwdot[CH3]/d[H2]
  J[17] -= dqdci; // dwdot[CH4]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[4];
  J[27] += dqdci; // dwdot[H2]/d[H]
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[29] += dqdci; // dwdot[CH3]/d[H]
  J[30] -= dqdci; // dwdot[CH4]/d[H]
  // d()/d[CH3]
  dqdci = -k_r * sc[1];
  J[40] += dqdci; // dwdot[H2]/d[CH3]
  J[41] -= dqdci; // dwdot[H]/d[CH3]
  J[42] += dqdci; // dwdot[CH3]/d[CH3]
  J[43] -= dqdci; // dwdot[CH4]/d[CH3]
  // d()/d[CH4]
  dqdci = +k_f * sc[2];
  J[53] += dqdci; // dwdot[H2]/d[CH4]
  J[54] -= dqdci; // dwdot[H]/d[CH4]
  J[55] += dqdci; // dwdot[CH3]/d[CH4]
  J[56] -= dqdci; // dwdot[CH4]/d[CH4]
  // d()/dT
  J[157] += dqdT; // dwdot[H2]/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[159] += dqdT; // dwdot[CH3]/dT
  J[160] -= dqdT; // dwdot[CH4]/dT

  // reaction 2: 2 CH3 <=> C2H6
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[3] * sc[3]);
  k_f = 6028000000000 * exp(0.1215 * logT - (2274.33727303233) * invT);
  dlnkfdT = 0.1215 * invT + (2274.33727303233) * invT2;
  // reverse
  phi_r = sc[10];
  Kc = refCinv * exp(2.000000 * g_RT[3] - g_RT[10]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[3]) + (h_RT[10]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= 2 * q; // CH3
  wdot[10] += q;    // C2H6
  // d()/d[CH3]
  dqdci = +k_f * 2.000000 * sc[3];
  J[42] += -2 * dqdci; // dwdot[CH3]/d[CH3]
  J[49] += dqdci;      // dwdot[C2H6]/d[CH3]
  // d()/d[C2H6]
  dqdci = -k_r;
  J[133] += -2 * dqdci; // dwdot[CH3]/d[C2H6]
  J[140] += dqdci;      // dwdot[C2H6]/d[C2H6]
  // d()/dT
  J[159] += -2 * dqdT; // dwdot[CH3]/dT
  J[166] += dqdT;      // dwdot[C2H6]/dT

  // reaction 3: 2 CH3 <=> C2H5 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[3] * sc[3]);
  k_f = 875200000 * exp(-0.9293 * logT - (52.3903392984072) * invT);
  dlnkfdT = -0.9293 * invT + (52.3903392984072) * invT2;
  // reverse
  phi_r = sc[2] * sc[9];
  Kc = exp(-g_RT[2] + 2.000000 * g_RT[3] - g_RT[9]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[3]) + (h_RT[2] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;     // H
  wdot[3] -= 2 * q; // CH3
  wdot[9] += q;     // C2H5
  // d()/d[H]
  dqdci = -k_r * sc[9];
  J[28] += dqdci;      // dwdot[H]/d[H]
  J[29] += -2 * dqdci; // dwdot[CH3]/d[H]
  J[35] += dqdci;      // dwdot[C2H5]/d[H]
  // d()/d[CH3]
  dqdci = +k_f * 2.000000 * sc[3];
  J[41] += dqdci;      // dwdot[H]/d[CH3]
  J[42] += -2 * dqdci; // dwdot[CH3]/d[CH3]
  J[48] += dqdci;      // dwdot[C2H5]/d[CH3]
  // d()/d[C2H5]
  dqdci = -k_r * sc[2];
  J[119] += dqdci;      // dwdot[H]/d[C2H5]
  J[120] += -2 * dqdci; // dwdot[CH3]/d[C2H5]
  J[126] += dqdci;      // dwdot[C2H5]/d[C2H5]
  // d()/dT
  J[158] += dqdT;      // dwdot[H]/dT
  J[159] += -2 * dqdT; // dwdot[CH3]/dT
  J[165] += dqdT;      // dwdot[C2H5]/dT

  // reaction 4: C2H6 + H <=> C2H5 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[10];
  k_f = 126.5 * exp(2.281 * logT - (5787.47274343287) * invT);
  dlnkfdT = 2.281 * invT + (5787.47274343287) * invT2;
  // reverse
  phi_r = sc[1] * sc[9];
  Kc = exp(-g_RT[1] + g_RT[2] - g_RT[9] + g_RT[10]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[10]) + (h_RT[1] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[9] += q;  // C2H5
  wdot[10] -= q; // C2H6
  // d()/d[H2]
  dqdci = -k_r * sc[9];
  J[14] += dqdci; // dwdot[H2]/d[H2]
  J[15] -= dqdci; // dwdot[H]/d[H2]
  J[22] += dqdci; // dwdot[C2H5]/d[H2]
  J[23] -= dqdci; // dwdot[C2H6]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[10];
  J[27] += dqdci; // dwdot[H2]/d[H]
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[35] += dqdci; // dwdot[C2H5]/d[H]
  J[36] -= dqdci; // dwdot[C2H6]/d[H]
  // d()/d[C2H5]
  dqdci = -k_r * sc[1];
  J[118] += dqdci; // dwdot[H2]/d[C2H5]
  J[119] -= dqdci; // dwdot[H]/d[C2H5]
  J[126] += dqdci; // dwdot[C2H5]/d[C2H5]
  J[127] -= dqdci; // dwdot[C2H6]/d[C2H5]
  // d()/d[C2H6]
  dqdci = +k_f * sc[2];
  J[131] += dqdci; // dwdot[H2]/d[C2H6]
  J[132] -= dqdci; // dwdot[H]/d[C2H6]
  J[139] += dqdci; // dwdot[C2H5]/d[C2H6]
  J[140] -= dqdci; // dwdot[C2H6]/d[C2H6]
  // d()/dT
  J[157] += dqdT; // dwdot[H2]/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[165] += dqdT; // dwdot[C2H5]/dT
  J[166] -= dqdT; // dwdot[C2H6]/dT

  // reaction 5: C2H4 + H <=> C2H5
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[8];
  k_f = 731500000 * exp(-0.6492 * logT - (13.2298836612139) * invT);
  dlnkfdT = -0.6492 * invT + (13.2298836612139) * invT2;
  // reverse
  phi_r = sc[9];
  Kc = refCinv * exp(g_RT[2] + g_RT[8] - g_RT[9]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[8]) + (h_RT[9]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[8] -= q; // C2H4
  wdot[9] += q; // C2H5
  // d()/d[H]
  dqdci = +k_f * sc[8];
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[34] -= dqdci; // dwdot[C2H4]/d[H]
  J[35] += dqdci; // dwdot[C2H5]/d[H]
  // d()/d[C2H4]
  dqdci = +k_f * sc[2];
  J[106] -= dqdci; // dwdot[H]/d[C2H4]
  J[112] -= dqdci; // dwdot[C2H4]/d[C2H4]
  J[113] += dqdci; // dwdot[C2H5]/d[C2H4]
  // d()/d[C2H5]
  dqdci = -k_r;
  J[119] -= dqdci; // dwdot[H]/d[C2H5]
  J[125] -= dqdci; // dwdot[C2H4]/d[C2H5]
  J[126] += dqdci; // dwdot[C2H5]/d[C2H5]
  // d()/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[164] -= dqdT; // dwdot[C2H4]/dT
  J[165] += dqdT; // dwdot[C2H5]/dT

  // reaction 6: C2H4 + H <=> C2H3 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[8];
  k_f = 24.64 * exp(1.851 * logT - (1875.03532980296) * invT);
  dlnkfdT = 1.851 * invT + (1875.03532980296) * invT2;
  // reverse
  phi_r = sc[1] * sc[7];
  Kc = exp(-g_RT[1] + g_RT[2] - g_RT[7] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[8]) + (h_RT[1] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // H2
  wdot[2] -= q; // H
  wdot[7] += q; // C2H3
  wdot[8] -= q; // C2H4
  // d()/d[H2]
  dqdci = -k_r * sc[7];
  J[14] += dqdci; // dwdot[H2]/d[H2]
  J[15] -= dqdci; // dwdot[H]/d[H2]
  J[20] += dqdci; // dwdot[C2H3]/d[H2]
  J[21] -= dqdci; // dwdot[C2H4]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[8];
  J[27] += dqdci; // dwdot[H2]/d[H]
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[33] += dqdci; // dwdot[C2H3]/d[H]
  J[34] -= dqdci; // dwdot[C2H4]/d[H]
  // d()/d[C2H3]
  dqdci = -k_r * sc[1];
  J[92] += dqdci; // dwdot[H2]/d[C2H3]
  J[93] -= dqdci; // dwdot[H]/d[C2H3]
  J[98] += dqdci; // dwdot[C2H3]/d[C2H3]
  J[99] -= dqdci; // dwdot[C2H4]/d[C2H3]
  // d()/d[C2H4]
  dqdci = +k_f * sc[2];
  J[105] += dqdci; // dwdot[H2]/d[C2H4]
  J[106] -= dqdci; // dwdot[H]/d[C2H4]
  J[111] += dqdci; // dwdot[C2H3]/d[C2H4]
  J[112] -= dqdci; // dwdot[C2H4]/d[C2H4]
  // d()/dT
  J[157] += dqdT; // dwdot[H2]/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[163] += dqdT; // dwdot[C2H3]/dT
  J[164] -= dqdT; // dwdot[C2H4]/dT

  // reaction 7: C2H2 + H <=> C2H + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[6];
  k_f = 3207000000 * exp(-0.0941 * logT - (29562.7763993308) * invT);
  dlnkfdT = -0.0941 * invT + (29562.7763993308) * invT2;
  // reverse
  phi_r = sc[1] * sc[5];
  Kc = exp(-g_RT[1] + g_RT[2] - g_RT[5] + g_RT[6]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[6]) + (h_RT[1] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // H2
  wdot[2] -= q; // H
  wdot[5] += q; // C2H
  wdot[6] -= q; // C2H2
  // d()/d[H2]
  dqdci = -k_r * sc[5];
  J[14] += dqdci; // dwdot[H2]/d[H2]
  J[15] -= dqdci; // dwdot[H]/d[H2]
  J[18] += dqdci; // dwdot[C2H]/d[H2]
  J[19] -= dqdci; // dwdot[C2H2]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[6];
  J[27] += dqdci; // dwdot[H2]/d[H]
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[31] += dqdci; // dwdot[C2H]/d[H]
  J[32] -= dqdci; // dwdot[C2H2]/d[H]
  // d()/d[C2H]
  dqdci = -k_r * sc[1];
  J[66] += dqdci; // dwdot[H2]/d[C2H]
  J[67] -= dqdci; // dwdot[H]/d[C2H]
  J[70] += dqdci; // dwdot[C2H]/d[C2H]
  J[71] -= dqdci; // dwdot[C2H2]/d[C2H]
  // d()/d[C2H2]
  dqdci = +k_f * sc[2];
  J[79] += dqdci; // dwdot[H2]/d[C2H2]
  J[80] -= dqdci; // dwdot[H]/d[C2H2]
  J[83] += dqdci; // dwdot[C2H]/d[C2H2]
  J[84] -= dqdci; // dwdot[C2H2]/d[C2H2]
  // d()/dT
  J[157] += dqdT; // dwdot[H2]/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[161] += dqdT; // dwdot[C2H]/dT
  J[162] -= dqdT; // dwdot[C2H2]/dT

  // reaction 8: C2H2 + H <=> C2H3
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[6];
  k_f = 86330000 * exp(-0.2163 * logT - (1321.78564942492) * invT);
  dlnkfdT = -0.2163 * invT + (1321.78564942492) * invT2;
  // reverse
  phi_r = sc[7];
  Kc = refCinv * exp(g_RT[2] + g_RT[6] - g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[6]) + (h_RT[7]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[6] -= q; // C2H2
  wdot[7] += q; // C2H3
  // d()/d[H]
  dqdci = +k_f * sc[6];
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[32] -= dqdci; // dwdot[C2H2]/d[H]
  J[33] += dqdci; // dwdot[C2H3]/d[H]
  // d()/d[C2H2]
  dqdci = +k_f * sc[2];
  J[80] -= dqdci; // dwdot[H]/d[C2H2]
  J[84] -= dqdci; // dwdot[C2H2]/d[C2H2]
  J[85] += dqdci; // dwdot[C2H3]/d[C2H2]
  // d()/d[C2H3]
  dqdci = -k_r;
  J[93] -= dqdci; // dwdot[H]/d[C2H3]
  J[97] -= dqdci; // dwdot[C2H2]/d[C2H3]
  J[98] += dqdci; // dwdot[C2H3]/d[C2H3]
  // d()/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[162] -= dqdT; // dwdot[C2H2]/dT
  J[163] += dqdT; // dwdot[C2H3]/dT

  // reaction 9: 3 C2H2 <=> C6H6
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[6] * sc[6];
  k_f = 0.07581 * exp(-0.001516 * logT - (2.90335810528822) * invT);
  dlnkfdT = -0.001516 * invT + (2.90335810528822) * invT2;
  // reverse
  phi_r = sc[11];
  Kc = exp(3.000000 * g_RT[6] - g_RT[11]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(3.000000 * h_RT[6]) + (h_RT[11]) + 2.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[6] -= 3 * q; // C2H2
  wdot[11] += q;    // C6H6
  // d()/d[C2H2]
  dqdci = +k_f * 3.000000 * sc[6] * sc[6];
  J[84] += -3 * dqdci; // dwdot[C2H2]/d[C2H2]
  J[89] += dqdci;      // dwdot[C6H6]/d[C2H2]
  // d()/d[C6H6]
  dqdci = -k_r;
  J[149] += -3 * dqdci; // dwdot[C2H2]/d[C6H6]
  J[154] += dqdci;      // dwdot[C6H6]/d[C6H6]
  // d()/dT
  J[162] += -3 * dqdT; // dwdot[C2H2]/dT
  J[167] += dqdT;      // dwdot[C6H6]/dT

  amrex::Real c_R[12], dcRdT[12], e_RT[12];
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
  for (int k = 0; k < 12; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[156 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 12; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 12; ++m) {
      dehmixdc += eh_RT[m] * J[k * 13 + m];
    }
    J[k * 13 + 12] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[168] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
#else
  amrex::Abort();
#endif
}

// compute an approx to the reaction Jacobian (for preconditioning)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT_SIMPLIFIED(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real* Tp, const int* HP)
{
  amrex::Real c[12];

  for (int k = 0; k < 12; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian_precond(J, c, *Tp, *HP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 12; k++) {
    J[156 + k] *= 1.e-6;
    J[k * 13 + 12] *= 1.e6;
  }
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int consP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 169; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[12];
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
  for (int k = 0; k < 12; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[12];
  gibbs(g_RT, T);

  // compute the species enthalpy
  amrex::Real h_RT[12];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[12];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 0: CH3 + H <=> CH4
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[3];
  k_f = 3.674e+17 * exp(-2.713 * logT - (23.0079704035475) * invT);
  dlnkfdT = -2.713 * invT + (23.0079704035475) * invT2;
  // reverse
  phi_r = sc[4];
  Kc = refCinv * exp(g_RT[2] + g_RT[3] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[3]) + (h_RT[4]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[3] -= q; // CH3
  wdot[4] += q; // CH4
  // d()/d[H]
  dqdci = +k_f * sc[3];
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[29] -= dqdci; // dwdot[CH3]/d[H]
  J[30] += dqdci; // dwdot[CH4]/d[H]
  // d()/d[CH3]
  dqdci = +k_f * sc[2];
  J[41] -= dqdci; // dwdot[H]/d[CH3]
  J[42] -= dqdci; // dwdot[CH3]/d[CH3]
  J[43] += dqdci; // dwdot[CH4]/d[CH3]
  // d()/d[CH4]
  dqdci = -k_r;
  J[54] -= dqdci; // dwdot[H]/d[CH4]
  J[55] -= dqdci; // dwdot[CH3]/d[CH4]
  J[56] += dqdci; // dwdot[CH4]/d[CH4]
  // d()/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[159] -= dqdT; // dwdot[CH3]/dT
  J[160] += dqdT; // dwdot[CH4]/dT

  // reaction 1: CH4 + H <=> CH3 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[4];
  k_f = 943.9 * exp(1.409 * logT - (3977.3841152395) * invT);
  dlnkfdT = 1.409 * invT + (3977.3841152395) * invT2;
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
  wdot[1] += q; // H2
  wdot[2] -= q; // H
  wdot[3] += q; // CH3
  wdot[4] -= q; // CH4
  // d()/d[H2]
  dqdci = -k_r * sc[3];
  J[14] += dqdci; // dwdot[H2]/d[H2]
  J[15] -= dqdci; // dwdot[H]/d[H2]
  J[16] += dqdci; // dwdot[CH3]/d[H2]
  J[17] -= dqdci; // dwdot[CH4]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[4];
  J[27] += dqdci; // dwdot[H2]/d[H]
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[29] += dqdci; // dwdot[CH3]/d[H]
  J[30] -= dqdci; // dwdot[CH4]/d[H]
  // d()/d[CH3]
  dqdci = -k_r * sc[1];
  J[40] += dqdci; // dwdot[H2]/d[CH3]
  J[41] -= dqdci; // dwdot[H]/d[CH3]
  J[42] += dqdci; // dwdot[CH3]/d[CH3]
  J[43] -= dqdci; // dwdot[CH4]/d[CH3]
  // d()/d[CH4]
  dqdci = +k_f * sc[2];
  J[53] += dqdci; // dwdot[H2]/d[CH4]
  J[54] -= dqdci; // dwdot[H]/d[CH4]
  J[55] += dqdci; // dwdot[CH3]/d[CH4]
  J[56] -= dqdci; // dwdot[CH4]/d[CH4]
  // d()/dT
  J[157] += dqdT; // dwdot[H2]/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[159] += dqdT; // dwdot[CH3]/dT
  J[160] -= dqdT; // dwdot[CH4]/dT

  // reaction 2: 2 CH3 <=> C2H6
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[3] * sc[3]);
  k_f = 6028000000000 * exp(0.1215 * logT - (2274.33727303233) * invT);
  dlnkfdT = 0.1215 * invT + (2274.33727303233) * invT2;
  // reverse
  phi_r = sc[10];
  Kc = refCinv * exp(2.000000 * g_RT[3] - g_RT[10]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[3]) + (h_RT[10]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= 2 * q; // CH3
  wdot[10] += q;    // C2H6
  // d()/d[CH3]
  dqdci = +k_f * 2.000000 * sc[3];
  J[42] += -2 * dqdci; // dwdot[CH3]/d[CH3]
  J[49] += dqdci;      // dwdot[C2H6]/d[CH3]
  // d()/d[C2H6]
  dqdci = -k_r;
  J[133] += -2 * dqdci; // dwdot[CH3]/d[C2H6]
  J[140] += dqdci;      // dwdot[C2H6]/d[C2H6]
  // d()/dT
  J[159] += -2 * dqdT; // dwdot[CH3]/dT
  J[166] += dqdT;      // dwdot[C2H6]/dT

  // reaction 3: 2 CH3 <=> C2H5 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[3] * sc[3]);
  k_f = 875200000 * exp(-0.9293 * logT - (52.3903392984072) * invT);
  dlnkfdT = -0.9293 * invT + (52.3903392984072) * invT2;
  // reverse
  phi_r = sc[2] * sc[9];
  Kc = exp(-g_RT[2] + 2.000000 * g_RT[3] - g_RT[9]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[3]) + (h_RT[2] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;     // H
  wdot[3] -= 2 * q; // CH3
  wdot[9] += q;     // C2H5
  // d()/d[H]
  dqdci = -k_r * sc[9];
  J[28] += dqdci;      // dwdot[H]/d[H]
  J[29] += -2 * dqdci; // dwdot[CH3]/d[H]
  J[35] += dqdci;      // dwdot[C2H5]/d[H]
  // d()/d[CH3]
  dqdci = +k_f * 2.000000 * sc[3];
  J[41] += dqdci;      // dwdot[H]/d[CH3]
  J[42] += -2 * dqdci; // dwdot[CH3]/d[CH3]
  J[48] += dqdci;      // dwdot[C2H5]/d[CH3]
  // d()/d[C2H5]
  dqdci = -k_r * sc[2];
  J[119] += dqdci;      // dwdot[H]/d[C2H5]
  J[120] += -2 * dqdci; // dwdot[CH3]/d[C2H5]
  J[126] += dqdci;      // dwdot[C2H5]/d[C2H5]
  // d()/dT
  J[158] += dqdT;      // dwdot[H]/dT
  J[159] += -2 * dqdT; // dwdot[CH3]/dT
  J[165] += dqdT;      // dwdot[C2H5]/dT

  // reaction 4: C2H6 + H <=> C2H5 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[10];
  k_f = 126.5 * exp(2.281 * logT - (5787.47274343287) * invT);
  dlnkfdT = 2.281 * invT + (5787.47274343287) * invT2;
  // reverse
  phi_r = sc[1] * sc[9];
  Kc = exp(-g_RT[1] + g_RT[2] - g_RT[9] + g_RT[10]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[10]) + (h_RT[1] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[9] += q;  // C2H5
  wdot[10] -= q; // C2H6
  // d()/d[H2]
  dqdci = -k_r * sc[9];
  J[14] += dqdci; // dwdot[H2]/d[H2]
  J[15] -= dqdci; // dwdot[H]/d[H2]
  J[22] += dqdci; // dwdot[C2H5]/d[H2]
  J[23] -= dqdci; // dwdot[C2H6]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[10];
  J[27] += dqdci; // dwdot[H2]/d[H]
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[35] += dqdci; // dwdot[C2H5]/d[H]
  J[36] -= dqdci; // dwdot[C2H6]/d[H]
  // d()/d[C2H5]
  dqdci = -k_r * sc[1];
  J[118] += dqdci; // dwdot[H2]/d[C2H5]
  J[119] -= dqdci; // dwdot[H]/d[C2H5]
  J[126] += dqdci; // dwdot[C2H5]/d[C2H5]
  J[127] -= dqdci; // dwdot[C2H6]/d[C2H5]
  // d()/d[C2H6]
  dqdci = +k_f * sc[2];
  J[131] += dqdci; // dwdot[H2]/d[C2H6]
  J[132] -= dqdci; // dwdot[H]/d[C2H6]
  J[139] += dqdci; // dwdot[C2H5]/d[C2H6]
  J[140] -= dqdci; // dwdot[C2H6]/d[C2H6]
  // d()/dT
  J[157] += dqdT; // dwdot[H2]/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[165] += dqdT; // dwdot[C2H5]/dT
  J[166] -= dqdT; // dwdot[C2H6]/dT

  // reaction 5: C2H4 + H <=> C2H5
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[8];
  k_f = 731500000 * exp(-0.6492 * logT - (13.2298836612139) * invT);
  dlnkfdT = -0.6492 * invT + (13.2298836612139) * invT2;
  // reverse
  phi_r = sc[9];
  Kc = refCinv * exp(g_RT[2] + g_RT[8] - g_RT[9]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[8]) + (h_RT[9]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[8] -= q; // C2H4
  wdot[9] += q; // C2H5
  // d()/d[H]
  dqdci = +k_f * sc[8];
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[34] -= dqdci; // dwdot[C2H4]/d[H]
  J[35] += dqdci; // dwdot[C2H5]/d[H]
  // d()/d[C2H4]
  dqdci = +k_f * sc[2];
  J[106] -= dqdci; // dwdot[H]/d[C2H4]
  J[112] -= dqdci; // dwdot[C2H4]/d[C2H4]
  J[113] += dqdci; // dwdot[C2H5]/d[C2H4]
  // d()/d[C2H5]
  dqdci = -k_r;
  J[119] -= dqdci; // dwdot[H]/d[C2H5]
  J[125] -= dqdci; // dwdot[C2H4]/d[C2H5]
  J[126] += dqdci; // dwdot[C2H5]/d[C2H5]
  // d()/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[164] -= dqdT; // dwdot[C2H4]/dT
  J[165] += dqdT; // dwdot[C2H5]/dT

  // reaction 6: C2H4 + H <=> C2H3 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[8];
  k_f = 24.64 * exp(1.851 * logT - (1875.03532980296) * invT);
  dlnkfdT = 1.851 * invT + (1875.03532980296) * invT2;
  // reverse
  phi_r = sc[1] * sc[7];
  Kc = exp(-g_RT[1] + g_RT[2] - g_RT[7] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[8]) + (h_RT[1] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // H2
  wdot[2] -= q; // H
  wdot[7] += q; // C2H3
  wdot[8] -= q; // C2H4
  // d()/d[H2]
  dqdci = -k_r * sc[7];
  J[14] += dqdci; // dwdot[H2]/d[H2]
  J[15] -= dqdci; // dwdot[H]/d[H2]
  J[20] += dqdci; // dwdot[C2H3]/d[H2]
  J[21] -= dqdci; // dwdot[C2H4]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[8];
  J[27] += dqdci; // dwdot[H2]/d[H]
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[33] += dqdci; // dwdot[C2H3]/d[H]
  J[34] -= dqdci; // dwdot[C2H4]/d[H]
  // d()/d[C2H3]
  dqdci = -k_r * sc[1];
  J[92] += dqdci; // dwdot[H2]/d[C2H3]
  J[93] -= dqdci; // dwdot[H]/d[C2H3]
  J[98] += dqdci; // dwdot[C2H3]/d[C2H3]
  J[99] -= dqdci; // dwdot[C2H4]/d[C2H3]
  // d()/d[C2H4]
  dqdci = +k_f * sc[2];
  J[105] += dqdci; // dwdot[H2]/d[C2H4]
  J[106] -= dqdci; // dwdot[H]/d[C2H4]
  J[111] += dqdci; // dwdot[C2H3]/d[C2H4]
  J[112] -= dqdci; // dwdot[C2H4]/d[C2H4]
  // d()/dT
  J[157] += dqdT; // dwdot[H2]/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[163] += dqdT; // dwdot[C2H3]/dT
  J[164] -= dqdT; // dwdot[C2H4]/dT

  // reaction 7: C2H2 + H <=> C2H + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[6];
  k_f = 3207000000 * exp(-0.0941 * logT - (29562.7763993308) * invT);
  dlnkfdT = -0.0941 * invT + (29562.7763993308) * invT2;
  // reverse
  phi_r = sc[1] * sc[5];
  Kc = exp(-g_RT[1] + g_RT[2] - g_RT[5] + g_RT[6]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[6]) + (h_RT[1] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // H2
  wdot[2] -= q; // H
  wdot[5] += q; // C2H
  wdot[6] -= q; // C2H2
  // d()/d[H2]
  dqdci = -k_r * sc[5];
  J[14] += dqdci; // dwdot[H2]/d[H2]
  J[15] -= dqdci; // dwdot[H]/d[H2]
  J[18] += dqdci; // dwdot[C2H]/d[H2]
  J[19] -= dqdci; // dwdot[C2H2]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[6];
  J[27] += dqdci; // dwdot[H2]/d[H]
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[31] += dqdci; // dwdot[C2H]/d[H]
  J[32] -= dqdci; // dwdot[C2H2]/d[H]
  // d()/d[C2H]
  dqdci = -k_r * sc[1];
  J[66] += dqdci; // dwdot[H2]/d[C2H]
  J[67] -= dqdci; // dwdot[H]/d[C2H]
  J[70] += dqdci; // dwdot[C2H]/d[C2H]
  J[71] -= dqdci; // dwdot[C2H2]/d[C2H]
  // d()/d[C2H2]
  dqdci = +k_f * sc[2];
  J[79] += dqdci; // dwdot[H2]/d[C2H2]
  J[80] -= dqdci; // dwdot[H]/d[C2H2]
  J[83] += dqdci; // dwdot[C2H]/d[C2H2]
  J[84] -= dqdci; // dwdot[C2H2]/d[C2H2]
  // d()/dT
  J[157] += dqdT; // dwdot[H2]/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[161] += dqdT; // dwdot[C2H]/dT
  J[162] -= dqdT; // dwdot[C2H2]/dT

  // reaction 8: C2H2 + H <=> C2H3
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[6];
  k_f = 86330000 * exp(-0.2163 * logT - (1321.78564942492) * invT);
  dlnkfdT = -0.2163 * invT + (1321.78564942492) * invT2;
  // reverse
  phi_r = sc[7];
  Kc = refCinv * exp(g_RT[2] + g_RT[6] - g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[6]) + (h_RT[7]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[6] -= q; // C2H2
  wdot[7] += q; // C2H3
  // d()/d[H]
  dqdci = +k_f * sc[6];
  J[28] -= dqdci; // dwdot[H]/d[H]
  J[32] -= dqdci; // dwdot[C2H2]/d[H]
  J[33] += dqdci; // dwdot[C2H3]/d[H]
  // d()/d[C2H2]
  dqdci = +k_f * sc[2];
  J[80] -= dqdci; // dwdot[H]/d[C2H2]
  J[84] -= dqdci; // dwdot[C2H2]/d[C2H2]
  J[85] += dqdci; // dwdot[C2H3]/d[C2H2]
  // d()/d[C2H3]
  dqdci = -k_r;
  J[93] -= dqdci; // dwdot[H]/d[C2H3]
  J[97] -= dqdci; // dwdot[C2H2]/d[C2H3]
  J[98] += dqdci; // dwdot[C2H3]/d[C2H3]
  // d()/dT
  J[158] -= dqdT; // dwdot[H]/dT
  J[162] -= dqdT; // dwdot[C2H2]/dT
  J[163] += dqdT; // dwdot[C2H3]/dT

  // reaction 9: 3 C2H2 <=> C6H6
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[6] * sc[6];
  k_f = 0.07581 * exp(-0.001516 * logT - (2.90335810528822) * invT);
  dlnkfdT = -0.001516 * invT + (2.90335810528822) * invT2;
  // reverse
  phi_r = sc[11];
  Kc = exp(3.000000 * g_RT[6] - g_RT[11]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(3.000000 * h_RT[6]) + (h_RT[11]) + 2.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[6] -= 3 * q; // C2H2
  wdot[11] += q;    // C6H6
  // d()/d[C2H2]
  dqdci = +k_f * 3.000000 * sc[6] * sc[6];
  J[84] += -3 * dqdci; // dwdot[C2H2]/d[C2H2]
  J[89] += dqdci;      // dwdot[C6H6]/d[C2H2]
  // d()/d[C6H6]
  dqdci = -k_r;
  J[149] += -3 * dqdci; // dwdot[C2H2]/d[C6H6]
  J[154] += dqdci;      // dwdot[C6H6]/d[C6H6]
  // d()/dT
  J[162] += -3 * dqdT; // dwdot[C2H2]/dT
  J[167] += dqdT;      // dwdot[C6H6]/dT

  amrex::Real c_R[12], dcRdT[12], e_RT[12];
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
  for (int k = 0; k < 12; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[156 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 12; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 12; ++m) {
      dehmixdc += eh_RT[m] * J[k * 13 + m];
    }
    J[k * 13 + 12] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[168] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
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
  amrex::Real c[12];

  for (int k = 0; k < 12; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian(J, c, *Tp, *consP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 12; k++) {
    J[156 + k] *= 1.e-6;
    J[k * 13 + 12] *= 1.e6;
  }
}

// Transport function declarations

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENIMC(int* LENIMC)
{
  *LENIMC = 50;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENRMC(int* LENRMC)
{
  *LENRMC = 3156;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNO(int* NO)
{
  *NO = 4;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKK(int* KK)
{
  *KK = 12;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLITE(int* NLITE)
{
  *NLITE = 2;
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
  WT[0] = 3.99500000E+01;
  WT[1] = 2.01600000E+00;
  WT[2] = 1.00800000E+00;
  WT[3] = 1.50350000E+01;
  WT[4] = 1.60430000E+01;
  WT[5] = 2.50300000E+01;
  WT[6] = 2.60380000E+01;
  WT[7] = 2.70460000E+01;
  WT[8] = 2.80540000E+01;
  WT[9] = 2.90620000E+01;
  WT[10] = 3.00700000E+01;
  WT[11] = 7.81140000E+01;
}

// the lennard-jones potential well depth eps/kb in K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetEPS(amrex::Real* EPS)
{
  EPS[0] = 1.36500000E+02;
  EPS[1] = 3.80000000E+01;
  EPS[2] = 1.45000000E+02;
  EPS[3] = 1.44000000E+02;
  EPS[4] = 1.41400000E+02;
  EPS[5] = 2.09000000E+02;
  EPS[6] = 2.09000000E+02;
  EPS[7] = 2.09000000E+02;
  EPS[8] = 2.80800000E+02;
  EPS[9] = 2.52300000E+02;
  EPS[10] = 2.52300000E+02;
  EPS[11] = 4.64800000E+02;
}

// the lennard-jones collision diameter in Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetSIG(amrex::Real* SIG)
{
  SIG[0] = 3.33000000E+00;
  SIG[1] = 2.92000000E+00;
  SIG[2] = 2.05000000E+00;
  SIG[3] = 3.80000000E+00;
  SIG[4] = 3.74600000E+00;
  SIG[5] = 4.10000000E+00;
  SIG[6] = 4.10000000E+00;
  SIG[7] = 4.10000000E+00;
  SIG[8] = 3.97100000E+00;
  SIG[9] = 4.30200000E+00;
  SIG[10] = 4.30200000E+00;
  SIG[11] = 5.29000000E+00;
}

// the dipole moment in Debye
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetDIP(amrex::Real* DIP)
{
  DIP[0] = 0.00000000E+00;
  DIP[1] = 0.00000000E+00;
  DIP[2] = 0.00000000E+00;
  DIP[3] = 0.00000000E+00;
  DIP[4] = 0.00000000E+00;
  DIP[5] = 0.00000000E+00;
  DIP[6] = 0.00000000E+00;
  DIP[7] = 0.00000000E+00;
  DIP[8] = 0.00000000E+00;
  DIP[9] = 0.00000000E+00;
  DIP[10] = 0.00000000E+00;
  DIP[11] = 0.00000000E+00;
}

// the polarizability in cubic Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPOL(amrex::Real* POL)
{
  POL[0] = 0.00000000E+00;
  POL[1] = 7.90000000E-01;
  POL[2] = 0.00000000E+00;
  POL[3] = 0.00000000E+00;
  POL[4] = 2.60000000E+00;
  POL[5] = 0.00000000E+00;
  POL[6] = 0.00000000E+00;
  POL[7] = 0.00000000E+00;
  POL[8] = 0.00000000E+00;
  POL[9] = 0.00000000E+00;
  POL[10] = 0.00000000E+00;
  POL[11] = 1.03200000E+01;
}

// the rotational relaxation collision number at 298 K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetZROT(amrex::Real* ZROT)
{
  ZROT[0] = 0.00000000E+00;
  ZROT[1] = 2.80000000E+02;
  ZROT[2] = 0.00000000E+00;
  ZROT[3] = 0.00000000E+00;
  ZROT[4] = 1.30000000E+01;
  ZROT[5] = 2.50000000E+00;
  ZROT[6] = 2.50000000E+00;
  ZROT[7] = 1.00000000E+00;
  ZROT[8] = 1.50000000E+00;
  ZROT[9] = 1.50000000E+00;
  ZROT[10] = 1.50000000E+00;
  ZROT[11] = 0.00000000E+00;
}

// 0: monoatomic, 1: linear, 2: nonlinear
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLIN(int* NLIN)
{
  NLIN[0] = 0;
  NLIN[1] = 1;
  NLIN[2] = 0;
  NLIN[3] = 2;
  NLIN[4] = 2;
  NLIN[5] = 1;
  NLIN[6] = 1;
  NLIN[7] = 2;
  NLIN[8] = 2;
  NLIN[9] = 2;
  NLIN[10] = 2;
  NLIN[11] = 2;
}

// Poly fits for the viscosities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFETA(amrex::Real* COFETA)
{
  COFETA[0] = -1.82621813E+01;
  COFETA[1] = 3.08577372E+00;
  COFETA[2] = -3.13545186E-01;
  COFETA[3] = 1.33929714E-02;
  COFETA[4] = -1.40163864E+01;
  COFETA[5] = 1.07258887E+00;
  COFETA[6] = -5.90183195E-02;
  COFETA[7] = 2.70099524E-03;
  COFETA[8] = -1.95693633E+01;
  COFETA[9] = 3.24443322E+00;
  COFETA[10] = -3.33105053E-01;
  COFETA[11] = 1.41964131E-02;
  COFETA[12] = -1.93944362E+01;
  COFETA[13] = 3.22309099E+00;
  COFETA[14] = -3.30444315E-01;
  COFETA[15] = 1.40859068E-02;
  COFETA[16] = -1.91946055E+01;
  COFETA[17] = 3.17252861E+00;
  COFETA[18] = -3.24185221E-01;
  COFETA[19] = 1.38277451E-02;
  COFETA[20] = -2.17905126E+01;
  COFETA[21] = 4.08495587E+00;
  COFETA[22] = -4.31526143E-01;
  COFETA[23] = 1.80295280E-02;
  COFETA[24] = -2.17707716E+01;
  COFETA[25] = 4.08495587E+00;
  COFETA[26] = -4.31526143E-01;
  COFETA[27] = 1.80295280E-02;
  COFETA[28] = -2.17517806E+01;
  COFETA[29] = 4.08495587E+00;
  COFETA[30] = -4.31526143E-01;
  COFETA[31] = 1.80295280E-02;
  COFETA[32] = -2.27723503E+01;
  COFETA[33] = 4.36449032E+00;
  COFETA[34] = -4.52534973E-01;
  COFETA[35] = 1.83577891E-02;
  COFETA[36] = -2.26516002E+01;
  COFETA[37] = 4.32291620E+00;
  COFETA[38] = -4.53357194E-01;
  COFETA[39] = 1.86266758E-02;
  COFETA[40] = -2.26345520E+01;
  COFETA[41] = 4.32291620E+00;
  COFETA[42] = -4.53357194E-01;
  COFETA[43] = 1.86266758E-02;
  COFETA[44] = -2.04567218E+01;
  COFETA[45] = 3.02966559E+00;
  COFETA[46] = -2.37626298E-01;
  COFETA[47] = 7.55708213E-03;
}

// Poly fits for the conductivities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFLAM(amrex::Real* COFLAM)
{
  COFLAM[0] = -2.39196110E+00;
  COFLAM[1] = 3.08577372E+00;
  COFLAM[2] = -3.13545186E-01;
  COFLAM[3] = 1.33929714E-02;
  COFLAM[4] = 1.15759774E+01;
  COFLAM[5] = -1.67302377E+00;
  COFLAM[6] = 3.16356982E-01;
  COFLAM[7] = -1.34571055E-02;
  COFLAM[8] = -1.94825747E-02;
  COFLAM[9] = 3.24443322E+00;
  COFLAM[10] = -3.33105053E-01;
  COFLAM[11] = 1.41964131E-02;
  COFLAM[12] = 5.67848538E+00;
  COFLAM[13] = -9.39424324E-01;
  COFLAM[14] = 3.61568847E-01;
  COFLAM[15] = -2.10729666E-02;
  COFLAM[16] = 6.51686604E+00;
  COFLAM[17] = -1.66213186E+00;
  COFLAM[18] = 5.14860177E-01;
  COFLAM[19] = -2.99960118E-02;
  COFLAM[20] = 1.22063731E+00;
  COFLAM[21] = 9.41096943E-01;
  COFLAM[22] = 8.03460177E-02;
  COFLAM[23] = -7.85155596E-03;
  COFLAM[24] = -1.14551106E+01;
  COFLAM[25] = 6.15097845E+00;
  COFLAM[26] = -6.25423721E-01;
  COFLAM[27] = 2.42715152E-02;
  COFLAM[28] = -7.36746966E+00;
  COFLAM[29] = 3.98647096E+00;
  COFLAM[30] = -2.65683436E-01;
  COFLAM[31] = 5.43389617E-03;
  COFLAM[32] = -1.10551979E+01;
  COFLAM[33] = 5.15487336E+00;
  COFLAM[34] = -3.76674333E-01;
  COFLAM[35] = 8.66254609E-03;
  COFLAM[36] = -8.73855773E+00;
  COFLAM[37] = 4.26084645E+00;
  COFLAM[38] = -2.62880854E-01;
  COFLAM[39] = 3.84734034E-03;
  COFLAM[40] = -1.03407395E+01;
  COFLAM[41] = 4.79019550E+00;
  COFLAM[42] = -3.14755548E-01;
  COFLAM[43] = 5.44545388E-03;
  COFLAM[44] = -2.90302855E+01;
  COFLAM[45] = 1.19575771E+01;
  COFLAM[46] = -1.25320889E+00;
  COFLAM[47] = 4.62052407E-02;
}

// Poly fits for the diffusion coefficients, dim NO*KK*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFD(amrex::Real* COFD)
{
  COFD[0] = -1.66628717E+01;
  COFD[1] = 3.81409115E+00;
  COFD[2] = -2.74349472E-01;
  COFD[3] = 1.16003544E-02;
  COFD[4] = -1.23245240E+01;
  COFD[5] = 2.72434624E+00;
  COFD[6] = -1.39891425E-01;
  COFD[7] = 6.07595720E-03;
  COFD[8] = -1.00738016E+01;
  COFD[9] = 2.06997694E+00;
  COFD[10] = -5.59541402E-02;
  COFD[11] = 2.50006701E-03;
  COFD[12] = -1.48965948E+01;
  COFD[13] = 3.87235061E+00;
  COFD[14] = -2.81353670E-01;
  COFD[15] = 1.18812044E-02;
  COFD[16] = -1.14093292E+01;
  COFD[17] = 2.74510463E+00;
  COFD[18] = -1.42059490E-01;
  COFD[19] = 6.14599532E-03;
  COFD[20] = -1.42068119E+01;
  COFD[21] = 3.93919304E+00;
  COFD[22] = -2.89498522E-01;
  COFD[23] = 1.22123407E-02;
  COFD[24] = -1.66450970E+01;
  COFD[25] = 3.86583801E+00;
  COFD[26] = -2.80574734E-01;
  COFD[27] = 1.18501372E-02;
  COFD[28] = -1.24846088E+01;
  COFD[29] = 2.73970291E+00;
  COFD[30] = -1.41360181E-01;
  COFD[31] = 6.11580523E-03;
  COFD[32] = -1.52083653E+01;
  COFD[33] = 3.93074587E+00;
  COFD[34] = -2.88461000E-01;
  COFD[35] = 1.21698290E-02;
  COFD[36] = -1.67452067E+01;
  COFD[37] = 3.92222903E+00;
  COFD[38] = -2.87413822E-01;
  COFD[39] = 1.21268730E-02;
  COFD[40] = -1.66048371E+01;
  COFD[41] = 3.84903314E+00;
  COFD[42] = -2.78572093E-01;
  COFD[43] = 1.17705788E-02;
  COFD[44] = -1.24688910E+01;
  COFD[45] = 2.74113804E+00;
  COFD[46] = -1.41842904E-01;
  COFD[47] = 6.14997445E-03;
  COFD[48] = -1.51289374E+01;
  COFD[49] = 3.90807300E+00;
  COFD[50] = -2.85669213E-01;
  COFD[51] = 1.20551365E-02;
  COFD[52] = -1.66848935E+01;
  COFD[53] = 3.89988730E+00;
  COFD[54] = -2.84665047E-01;
  COFD[55] = 1.20140370E-02;
  COFD[56] = -1.66362939E+01;
  COFD[57] = 3.88210262E+00;
  COFD[58] = -2.82525119E-01;
  COFD[59] = 1.19281432E-02;
  COFD[60] = -1.79802297E+01;
  COFD[61] = 4.24190795E+00;
  COFD[62] = -3.25395075E-01;
  COFD[63] = 1.36317650E-02;
  COFD[64] = -1.31797238E+01;
  COFD[65] = 2.93540484E+00;
  COFD[66] = -1.64051615E-01;
  COFD[67] = 6.97863026E-03;
  COFD[68] = -1.64202002E+01;
  COFD[69] = 4.31189163E+00;
  COFD[70] = -3.33731775E-01;
  COFD[71] = 1.39630543E-02;
  COFD[72] = -1.80347108E+01;
  COFD[73] = 4.30427702E+00;
  COFD[74] = -3.32832756E-01;
  COFD[75] = 1.39276609E-02;
  COFD[76] = -1.79823838E+01;
  COFD[77] = 4.28398102E+00;
  COFD[78] = -3.30432012E-01;
  COFD[79] = 1.38329696E-02;
  COFD[80] = -1.91886979E+01;
  COFD[81] = 4.60477462E+00;
  COFD[82] = -3.65745983E-01;
  COFD[83] = 1.51175537E-02;
  COFD[84] = -1.79922739E+01;
  COFD[85] = 4.24190795E+00;
  COFD[86] = -3.25395075E-01;
  COFD[87] = 1.36317650E-02;
  COFD[88] = -1.31811687E+01;
  COFD[89] = 2.93540484E+00;
  COFD[90] = -1.64051615E-01;
  COFD[91] = 6.97863026E-03;
  COFD[92] = -1.64209501E+01;
  COFD[93] = 4.31189163E+00;
  COFD[94] = -3.33731775E-01;
  COFD[95] = 1.39630543E-02;
  COFD[96] = -1.80420278E+01;
  COFD[97] = 4.30427702E+00;
  COFD[98] = -3.32832756E-01;
  COFD[99] = 1.39276609E-02;
  COFD[100] = -1.79900021E+01;
  COFD[101] = 4.28398102E+00;
  COFD[102] = -3.30432012E-01;
  COFD[103] = 1.38329696E-02;
  COFD[104] = -1.91984709E+01;
  COFD[105] = 4.60477462E+00;
  COFD[106] = -3.65745983E-01;
  COFD[107] = 1.51175537E-02;
  COFD[108] = -1.92084388E+01;
  COFD[109] = 4.60477462E+00;
  COFD[110] = -3.65745983E-01;
  COFD[111] = 1.51175537E-02;
  COFD[112] = -1.80036850E+01;
  COFD[113] = 4.24190795E+00;
  COFD[114] = -3.25395075E-01;
  COFD[115] = 1.36317650E-02;
  COFD[116] = -1.31825097E+01;
  COFD[117] = 2.93540484E+00;
  COFD[118] = -1.64051615E-01;
  COFD[119] = 6.97863026E-03;
  COFD[120] = -1.64216451E+01;
  COFD[121] = 4.31189163E+00;
  COFD[122] = -3.33731775E-01;
  COFD[123] = 1.39630543E-02;
  COFD[124] = -1.80488962E+01;
  COFD[125] = 4.30427702E+00;
  COFD[126] = -3.32832756E-01;
  COFD[127] = 1.39276609E-02;
  COFD[128] = -1.79971574E+01;
  COFD[129] = 4.28398102E+00;
  COFD[130] = -3.30432012E-01;
  COFD[131] = 1.38329696E-02;
  COFD[132] = -1.92076889E+01;
  COFD[133] = 4.60477462E+00;
  COFD[134] = -3.65745983E-01;
  COFD[135] = 1.51175537E-02;
  COFD[136] = -1.92178442E+01;
  COFD[137] = 4.60477462E+00;
  COFD[138] = -3.65745983E-01;
  COFD[139] = 1.51175537E-02;
  COFD[140] = -1.92274299E+01;
  COFD[141] = 4.60477462E+00;
  COFD[142] = -3.65745983E-01;
  COFD[143] = 1.51175537E-02;
  COFD[144] = -1.87796248E+01;
  COFD[145] = 4.50615410E+00;
  COFD[146] = -3.55229692E-01;
  COFD[147] = 1.47498685E-02;
  COFD[148] = -1.38547681E+01;
  COFD[149] = 3.19493350E+00;
  COFD[150] = -1.96874917E-01;
  COFD[151] = 8.36487936E-03;
  COFD[152] = -1.71170365E+01;
  COFD[153] = 4.54870786E+00;
  COFD[154] = -3.59689602E-01;
  COFD[155] = 1.49021258E-02;
  COFD[156] = -1.87638319E+01;
  COFD[157] = 4.54305237E+00;
  COFD[158] = -3.59077943E-01;
  COFD[159] = 1.48803637E-02;
  COFD[160] = -1.87272828E+01;
  COFD[161] = 4.52952998E+00;
  COFD[162] = -3.57652782E-01;
  COFD[163] = 1.48313625E-02;
  COFD[164] = -1.98177805E+01;
  COFD[165] = 4.78964109E+00;
  COFD[166] = -3.83764078E-01;
  COFD[167] = 1.56720731E-02;
  COFD[168] = -1.98281161E+01;
  COFD[169] = 4.78964109E+00;
  COFD[170] = -3.83764078E-01;
  COFD[171] = 1.56720731E-02;
  COFD[172] = -1.98378755E+01;
  COFD[173] = 4.78964109E+00;
  COFD[174] = -3.83764078E-01;
  COFD[175] = 1.56720731E-02;
  COFD[176] = -2.02819040E+01;
  COFD[177] = 4.89993603E+00;
  COFD[178] = -3.91106887E-01;
  COFD[179] = 1.57323143E-02;
  COFD[180] = -1.86367575E+01;
  COFD[181] = 4.43083272E+00;
  COFD[182] = -3.47252556E-01;
  COFD[183] = 1.44736342E-02;
  COFD[184] = -1.36619307E+01;
  COFD[185] = 3.08793502E+00;
  COFD[186] = -1.83196237E-01;
  COFD[187] = 7.78096956E-03;
  COFD[188] = -1.69847109E+01;
  COFD[189] = 4.47462978E+00;
  COFD[190] = -3.51930069E-01;
  COFD[191] = 1.46373700E-02;
  COFD[192] = -1.86152644E+01;
  COFD[193] = 4.46964399E+00;
  COFD[194] = -3.51397975E-01;
  COFD[195] = 1.46187564E-02;
  COFD[196] = -1.85812108E+01;
  COFD[197] = 4.45654764E+00;
  COFD[198] = -3.50002949E-01;
  COFD[199] = 1.45701021E-02;
  COFD[200] = -1.97264644E+01;
  COFD[201] = 4.74481316E+00;
  COFD[202] = -3.80433916E-01;
  COFD[203] = 1.56213371E-02;
  COFD[204] = -1.97369736E+01;
  COFD[205] = 4.74481316E+00;
  COFD[206] = -3.80433916E-01;
  COFD[207] = 1.56213371E-02;
  COFD[208] = -1.97469003E+01;
  COFD[209] = 4.74481316E+00;
  COFD[210] = -3.80433916E-01;
  COFD[211] = 1.56213371E-02;
  COFD[212] = -2.02665002E+01;
  COFD[213] = 4.88760634E+00;
  COFD[214] = -3.92352054E-01;
  COFD[215] = 1.58907546E-02;
  COFD[216] = -2.01447733E+01;
  COFD[217] = 4.83021397E+00;
  COFD[218] = -3.87205755E-01;
  COFD[219] = 1.57527665E-02;
  COFD[220] = -1.86465555E+01;
  COFD[221] = 4.43083272E+00;
  COFD[222] = -3.47252556E-01;
  COFD[223] = 1.44736342E-02;
  COFD[224] = -1.36630192E+01;
  COFD[225] = 3.08793502E+00;
  COFD[226] = -1.83196237E-01;
  COFD[227] = 7.78096956E-03;
  COFD[228] = -1.69852730E+01;
  COFD[229] = 4.47462978E+00;
  COFD[230] = -3.51930069E-01;
  COFD[231] = 1.46373700E-02;
  COFD[232] = -1.86210120E+01;
  COFD[233] = 4.46964399E+00;
  COFD[234] = -3.51397975E-01;
  COFD[235] = 1.46187564E-02;
  COFD[236] = -1.85872082E+01;
  COFD[237] = 4.45654764E+00;
  COFD[238] = -3.50002949E-01;
  COFD[239] = 1.45701021E-02;
  COFD[240] = -1.97342809E+01;
  COFD[241] = 4.74481316E+00;
  COFD[242] = -3.80433916E-01;
  COFD[243] = 1.56213371E-02;
  COFD[244] = -1.97449575E+01;
  COFD[245] = 4.74481316E+00;
  COFD[246] = -3.80433916E-01;
  COFD[247] = 1.56213371E-02;
  COFD[248] = -1.97550456E+01;
  COFD[249] = 4.74481316E+00;
  COFD[250] = -3.80433916E-01;
  COFD[251] = 1.56213371E-02;
  COFD[252] = -2.02748013E+01;
  COFD[253] = 4.88760634E+00;
  COFD[254] = -3.92352054E-01;
  COFD[255] = 1.58907546E-02;
  COFD[256] = -2.01532247E+01;
  COFD[257] = 4.83021397E+00;
  COFD[258] = -3.87205755E-01;
  COFD[259] = 1.57527665E-02;
  COFD[260] = -2.01618215E+01;
  COFD[261] = 4.83021397E+00;
  COFD[262] = -3.87205755E-01;
  COFD[263] = 1.57527665E-02;
  COFD[264] = -2.04398542E+01;
  COFD[265] = 4.82790049E+00;
  COFD[266] = -3.86974240E-01;
  COFD[267] = 1.57453039E-02;
  COFD[268] = -1.57669057E+01;
  COFD[269] = 3.75911914E+00;
  COFD[270] = -2.67666258E-01;
  COFD[271] = 1.13292709E-02;
  COFD[272] = -1.86084376E+01;
  COFD[273] = 4.86520268E+00;
  COFD[274] = -3.90548600E-01;
  COFD[275] = 1.58532006E-02;
  COFD[276] = -2.02921273E+01;
  COFD[277] = 4.86145474E+00;
  COFD[278] = -3.90209646E-01;
  COFD[279] = 1.58439565E-02;
  COFD[280] = -2.02698593E+01;
  COFD[281] = 4.85090297E+00;
  COFD[282] = -3.89224118E-01;
  COFD[283] = 1.58154406E-02;
  COFD[284] = -2.09390484E+01;
  COFD[285] = 4.90824670E+00;
  COFD[286] = -3.86434599E-01;
  COFD[287] = 1.53202865E-02;
  COFD[288] = -2.09539267E+01;
  COFD[289] = 4.90824670E+00;
  COFD[290] = -3.86434599E-01;
  COFD[291] = 1.53202865E-02;
  COFD[292] = -2.09681019E+01;
  COFD[293] = 4.90824670E+00;
  COFD[294] = -3.86434599E-01;
  COFD[295] = 1.53202865E-02;
  COFD[296] = -2.08795584E+01;
  COFD[297] = 4.78406777E+00;
  COFD[298] = -3.60920049E-01;
  COFD[299] = 1.38778606E-02;
  COFD[300] = -2.10186305E+01;
  COFD[301] = 4.84151422E+00;
  COFD[302] = -3.71885985E-01;
  COFD[303] = 1.44778125E-02;
  COFD[304] = -2.10309982E+01;
  COFD[305] = 4.84151422E+00;
  COFD[306] = -3.71885985E-01;
  COFD[307] = 1.44778125E-02;
  COFD[308] = -2.03776723E+01;
  COFD[309] = 4.16860900E+00;
  COFD[310] = -2.62223932E-01;
  COFD[311] = 8.94830624E-03;
}

// List of specs with small weight, dim NLITE
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKTDIF(int* KTDIF)
{
  KTDIF[0] = 1;
  KTDIF[1] = 2;
}

// Poly fits for thermal diff ratios, dim NO*NLITE*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFTD(amrex::Real* COFTD)
{
  COFTD[0] = 4.15292420E-01;
  COFTD[1] = 7.83414732E-05;
  COFTD[2] = -2.03098199E-08;
  COFTD[3] = 1.74179759E-12;
  COFTD[4] = 0.00000000E+00;
  COFTD[5] = 0.00000000E+00;
  COFTD[6] = 0.00000000E+00;
  COFTD[7] = 0.00000000E+00;
  COFTD[8] = -1.50897267E-01;
  COFTD[9] = -3.16439590E-05;
  COFTD[10] = 8.52440436E-09;
  COFTD[11] = -7.55367996E-13;
  COFTD[12] = 3.46248652E-01;
  COFTD[13] = 7.17439580E-05;
  COFTD[14] = -1.92497800E-08;
  COFTD[15] = 1.70031375E-12;
  COFTD[16] = 3.53833290E-01;
  COFTD[17] = 7.10241222E-05;
  COFTD[18] = -1.88485090E-08;
  COFTD[19] = 1.64986380E-12;
  COFTD[20] = 3.44210345E-01;
  COFTD[21] = 1.30486152E-04;
  COFTD[22] = -3.95806289E-08;
  COFTD[23] = 3.76822917E-12;
  COFTD[24] = 3.46377141E-01;
  COFTD[25] = 1.31307559E-04;
  COFTD[26] = -3.98297880E-08;
  COFTD[27] = 3.79195009E-12;
  COFTD[28] = 3.48393629E-01;
  COFTD[29] = 1.32071987E-04;
  COFTD[30] = -4.00616633E-08;
  COFTD[31] = 3.81402551E-12;
  COFTD[32] = 3.09260953E-01;
  COFTD[33] = 1.80769305E-04;
  COFTD[34] = -5.64102126E-08;
  COFTD[35] = 5.39257779E-12;
  COFTD[36] = 3.26446289E-01;
  COFTD[37] = 1.63628475E-04;
  COFTD[38] = -5.07308003E-08;
  COFTD[39] = 4.85318626E-12;
  COFTD[40] = 3.27975172E-01;
  COFTD[41] = 1.64394814E-04;
  COFTD[42] = -5.09683936E-08;
  COFTD[43] = 4.87591574E-12;
  COFTD[44] = 2.49371237E-01;
  COFTD[45] = 2.96138338E-04;
  COFTD[46] = -9.23707914E-08;
  COFTD[47] = 8.67703811E-12;
  COFTD[48] = 2.27681902E-01;
  COFTD[49] = 3.18845952E-04;
  COFTD[50] = -9.90253137E-08;
  COFTD[51] = 9.25184710E-12;
  COFTD[52] = 1.50897267E-01;
  COFTD[53] = 3.16439590E-05;
  COFTD[54] = -8.52440436E-09;
  COFTD[55] = 7.55367996E-13;
  COFTD[56] = 0.00000000E+00;
  COFTD[57] = 0.00000000E+00;
  COFTD[58] = 0.00000000E+00;
  COFTD[59] = 0.00000000E+00;
  COFTD[60] = 1.99749063E-01;
  COFTD[61] = 3.02793903E-04;
  COFTD[62] = -9.38174361E-08;
  COFTD[63] = 8.74222105E-12;
  COFTD[64] = 2.04762261E-01;
  COFTD[65] = 3.02081994E-04;
  COFTD[66] = -9.36745391E-08;
  COFTD[67] = 8.73678923E-12;
  COFTD[68] = 1.38545056E-01;
  COFTD[69] = 3.86844230E-04;
  COFTD[70] = -1.17344887E-07;
  COFTD[71] = 1.07245880E-11;
  COFTD[72] = 1.38978397E-01;
  COFTD[73] = 3.88054201E-04;
  COFTD[74] = -1.17711918E-07;
  COFTD[75] = 1.07581323E-11;
  COFTD[76] = 1.39380598E-01;
  COFTD[77] = 3.89177222E-04;
  COFTD[78] = -1.18052574E-07;
  COFTD[79] = 1.07892662E-11;
  COFTD[80] = 8.19828702E-02;
  COFTD[81] = 4.37579874E-04;
  COFTD[82] = -1.29925235E-07;
  COFTD[83] = 1.16814863E-11;
  COFTD[84] = 1.03028071E-01;
  COFTD[85] = 4.22339076E-04;
  COFTD[86] = -1.26428074E-07;
  COFTD[87] = 1.14358729E-11;
  COFTD[88] = 1.03268208E-01;
  COFTD[89] = 4.23323460E-04;
  COFTD[90] = -1.26722751E-07;
  COFTD[91] = 1.14625275E-11;
  COFTD[92] = -1.06910001E-02;
  COFTD[93] = 5.20253935E-04;
  COFTD[94] = -1.47716443E-07;
  COFTD[95] = 1.28814197E-11;
}

// compute the critical parameters for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS(
  amrex::Real* Tci, amrex::Real* ai, amrex::Real* bi, amrex::Real* acentric_i)
{

  amrex::Real EPS[12];
  amrex::Real SIG[12];
  amrex::Real wt[12];
  amrex::Real Rcst = 83.144598; // in bar [CGS] !
  amrex::Real avogadro = 6.02214199e23;
  amrex::Real boltzmann = 1.3806503e-16; // we work in CGS

  egtransetEPS(EPS);
  egtransetSIG(SIG);
  get_mw(wt);

  // species 0: Ar
  Tci[0] = 1.316 * EPS[0];
  ai[0] = (5.55 * avogadro * avogadro * EPS[0] * boltzmann * 1e-24 * SIG[0] *
           SIG[0] * SIG[0]) /
          (wt[0] * wt[0]);
  bi[0] = 0.855 * avogadro * 1e-24 * SIG[0] * SIG[0] * SIG[0] / (wt[0]);
  acentric_i[0] = 0.0;

  // species 1: H2
  // Imported from NIST
  Tci[1] = 33.145000;
  ai[1] = 1e6 * 0.42748 * Rcst * Rcst * Tci[1] * Tci[1] /
          (2.015880 * 2.015880 * 12.964000);
  bi[1] = 0.08664 * Rcst * Tci[1] / (2.015880 * 12.964000);
  acentric_i[1] = -0.219000;

  // species 2: H
  Tci[2] = 1.316 * EPS[2];
  ai[2] = (5.55 * avogadro * avogadro * EPS[2] * boltzmann * 1e-24 * SIG[2] *
           SIG[2] * SIG[2]) /
          (wt[2] * wt[2]);
  bi[2] = 0.855 * avogadro * 1e-24 * SIG[2] * SIG[2] * SIG[2] / (wt[2]);
  acentric_i[2] = 0.0;

  // species 3: CH3
  Tci[3] = 1.316 * EPS[3];
  ai[3] = (5.55 * avogadro * avogadro * EPS[3] * boltzmann * 1e-24 * SIG[3] *
           SIG[3] * SIG[3]) /
          (wt[3] * wt[3]);
  bi[3] = 0.855 * avogadro * 1e-24 * SIG[3] * SIG[3] * SIG[3] / (wt[3]);
  acentric_i[3] = 0.0;

  // species 4: CH4
  // Imported from NIST
  Tci[4] = 190.560000;
  ai[4] = 1e6 * 0.42748 * Rcst * Rcst * Tci[4] * Tci[4] /
          (16.043030 * 16.043030 * 45.990000);
  bi[4] = 0.08664 * Rcst * Tci[4] / (16.043030 * 45.990000);
  acentric_i[4] = 0.011000;

  // species 5: C2H
  Tci[5] = 1.316 * EPS[5];
  ai[5] = (5.55 * avogadro * avogadro * EPS[5] * boltzmann * 1e-24 * SIG[5] *
           SIG[5] * SIG[5]) /
          (wt[5] * wt[5]);
  bi[5] = 0.855 * avogadro * 1e-24 * SIG[5] * SIG[5] * SIG[5] / (wt[5]);
  acentric_i[5] = 0.0;

  // species 6: C2H2
  // Imported from NIST
  Tci[6] = 308.300000;
  ai[6] = 1e6 * 0.42748 * Rcst * Rcst * Tci[6] * Tci[6] /
          (26.038000 * 26.038000 * 61.140000);
  bi[6] = 0.08664 * Rcst * Tci[6] / (26.038000 * 61.140000);
  acentric_i[6] = 0.189000;

  // species 7: C2H3
  Tci[7] = 1.316 * EPS[7];
  ai[7] = (5.55 * avogadro * avogadro * EPS[7] * boltzmann * 1e-24 * SIG[7] *
           SIG[7] * SIG[7]) /
          (wt[7] * wt[7]);
  bi[7] = 0.855 * avogadro * 1e-24 * SIG[7] * SIG[7] * SIG[7] / (wt[7]);
  acentric_i[7] = 0.0;

  // species 8: C2H4
  // Imported from NIST
  Tci[8] = 282.340000;
  ai[8] = 1e6 * 0.42748 * Rcst * Rcst * Tci[8] * Tci[8] /
          (28.054000 * 28.054000 * 50.410000);
  bi[8] = 0.08664 * Rcst * Tci[8] / (28.054000 * 50.410000);
  acentric_i[8] = 0.087000;

  // species 9: C2H5
  Tci[9] = 1.316 * EPS[9];
  ai[9] = (5.55 * avogadro * avogadro * EPS[9] * boltzmann * 1e-24 * SIG[9] *
           SIG[9] * SIG[9]) /
          (wt[9] * wt[9]);
  bi[9] = 0.855 * avogadro * 1e-24 * SIG[9] * SIG[9] * SIG[9] / (wt[9]);
  acentric_i[9] = 0.0;

  // species 10: C2H6
  // Imported from NIST
  Tci[10] = 305.320000;
  ai[10] = 1e6 * 0.42748 * Rcst * Rcst * Tci[10] * Tci[10] /
           (30.070120 * 30.070120 * 48.720000);
  bi[10] = 0.08664 * Rcst * Tci[10] / (30.070120 * 48.720000);
  acentric_i[10] = 0.099000;

  // species 11: C6H6
  Tci[11] = 1.316 * EPS[11];
  ai[11] = (5.55 * avogadro * avogadro * EPS[11] * boltzmann * 1e-24 * SIG[11] *
            SIG[11] * SIG[11]) /
           (wt[11] * wt[11]);
  bi[11] = 0.855 * avogadro * 1e-24 * SIG[11] * SIG[11] * SIG[11] / (wt[11]);
  acentric_i[11] = 0.0;
}

// compute the critical parameter quantities for each species for SRK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS_SRK(
  amrex::Real* sqrtOneOverTc,
  amrex::Real* sqrtAsti,
  amrex::Real* Bi,
  amrex::Real* Fomega)
{

  // species 0: Ar
  // Computed from Lennard-Jones
  sqrtOneOverTc[0] = 7.4611492866627e-02;
  sqrtAsti[0] = 2.9624785926633e+04;
  Bi[0] = 4.7591897732266e-01;
  Fomega[0] = 4.8508000000000e-01;

  // species 1: H2
  // Imported from NIST
  sqrtOneOverTc[1] = 1.7369646834812e-01;
  sqrtAsti[1] = 2.4824229305911e+05;
  Bi[1] = 9.1362254169923e+00;
  Fomega[1] = 1.3798618890700e-01;

  // species 2: H
  // Computed from Lennard-Jones
  sqrtOneOverTc[2] = 7.2391579785311e-02;
  sqrtAsti[2] = 5.8451221185793e+05;
  Bi[2] = 4.4006634563410e+00;
  Fomega[2] = 4.8508000000000e-01;

  // species 3: CH3
  // Computed from Lennard-Jones
  sqrtOneOverTc[3] = 7.2642504557751e-02;
  sqrtAsti[3] = 9.8558148656101e+04;
  Bi[3] = 1.8791630452967e+00;
  Fomega[3] = 4.8508000000000e-01;

  // species 4: CH4
  // Imported from NIST
  sqrtOneOverTc[4] = 7.2440948474968e-02;
  sqrtAsti[4] = 9.5215046085708e+04;
  Bi[4] = 1.8605203320660e+00;
  Fomega[4] = 5.0213035482700e-01;

  // species 5: C2H
  // Computed from Lennard-Jones
  sqrtOneOverTc[5] = 6.0297445312864e-02;
  sqrtAsti[5] = 7.9933238165885e+04;
  Bi[5] = 1.4177766724704e+00;
  Fomega[5] = 4.8508000000000e-01;

  // species 6: C2H2
  // Imported from NIST
  sqrtOneOverTc[6] = 5.6952558345025e-02;
  sqrtAsti[6] = 8.2318059295808e+04;
  Bi[6] = 1.3950602011188e+00;
  Fomega[6] = 7.7293553202700e-01;

  // species 7: C2H3
  // Computed from Lennard-Jones
  sqrtOneOverTc[7] = 6.0297445312864e-02;
  sqrtAsti[7] = 7.3975040719223e+04;
  Bi[7] = 1.3120960627055e+00;
  Fomega[7] = 4.8508000000000e-01;

  // species 8: C2H4
  // Imported from NIST
  sqrtOneOverTc[8] = 5.9513267432530e-02;
  sqrtAsti[8] = 7.7056842134487e+04;
  Bi[8] = 1.4381805192666e+00;
  Fomega[8] = 6.1893034120300e-01;

  // species 9: C2H5
  // Computed from Lennard-Jones
  sqrtOneOverTc[9] = 5.4879915006830e-02;
  sqrtAsti[9] = 8.1297663542693e+04;
  Bi[9] = 1.4105965766665e+00;
  Fomega[9] = 4.8508000000000e-01;

  // species 10: C2H6
  // Imported from NIST
  sqrtOneOverTc[10] = 5.7229819101749e-02;
  sqrtAsti[10] = 7.9078494187583e+04;
  Bi[10] = 1.5012924285481e+00;
  Fomega[10] = 6.3721234098700e-01;

  // species 11: C6H6
  // Computed from Lennard-Jones
  sqrtOneOverTc[11] = 4.0433282083300e-02;
  sqrtAsti[11] = 5.5979242684102e+04;
  Bi[11] = 9.7578748676763e-01;
  Fomega[11] = 4.8508000000000e-01;
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
