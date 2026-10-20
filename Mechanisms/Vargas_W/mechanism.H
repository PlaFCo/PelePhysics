#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/* Elements
0  H
1  C
*/

// Species
#define H2_ID 0
#define H_ID 1
#define CH2_ID 2
#define CH3_ID 3
#define CH4_ID 4
#define C2H_ID 5
#define C2H2_ID 6
#define C2H3_ID 7
#define C2H4_ID 8
#define C2H5_ID 9
#define C2H6_ID 10
#define C_s_ID 11

#define NUM_GAS_ELEMENTS 2   // Elements in the homogeneous phase
#define NUM_GAS_SPECIES 12   // Species in the homogeneous phase
#define NUM_GAS_REACTIONS 14 // Reactions in the homogeneous phase

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
  mm = 2;
  kk = 12;
  ii = 14;
  nfit = -1; // Why do you need this anyway ?
}

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[12] = {
  4.9603174603174605e-01, // H2
  9.9206349206349209e-01, // H
  7.1291081485706148e-02, // CH2
  6.6511473229132020e-02, // CH3
  6.2332481456086768e-02, // CH4
  3.9952057530962849e-02, // C2H
  3.8405407481373381e-02, // C2H2
  3.6974044220956889e-02, // C2H3
  3.5645540742853074e-02, // C2H4
  3.4409194136673323e-02, // C2H5
  3.3255736614566010e-02, // C2H6
  8.3257014403463503e-02, // C(s)
};
#endif
const amrex::Real h_global_imw[12] = {
  4.9603174603174605e-01, // H2
  9.9206349206349209e-01, // H
  7.1291081485706148e-02, // CH2
  6.6511473229132020e-02, // CH3
  6.2332481456086768e-02, // CH4
  3.9952057530962849e-02, // C2H
  3.8405407481373381e-02, // C2H2
  3.6974044220956889e-02, // C2H3
  3.5645540742853074e-02, // C2H4
  3.4409194136673323e-02, // C2H5
  3.3255736614566010e-02, // C2H6
  8.3257014403463503e-02, // C(s)
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[12] = {
  2.0160000000000000e+00, // H2
  1.0080000000000000e+00, // H
  1.4026999999999999e+01, // CH2
  1.5035000000000000e+01, // CH3
  1.6042999999999999e+01, // CH4
  2.5029999999999998e+01, // C2H
  2.6037999999999997e+01, // C2H2
  2.7045999999999999e+01, // C2H3
  2.8053999999999998e+01, // C2H4
  2.9061999999999998e+01, // C2H5
  3.0070000000000000e+01, // C2H6
  1.2010999999999999e+01, // C(s)
};
#endif
const amrex::Real h_global_mw[12] = {
  2.0160000000000000e+00, // H2
  1.0080000000000000e+00, // H
  1.4026999999999999e+01, // CH2
  1.5035000000000000e+01, // CH3
  1.6042999999999999e+01, // CH4
  2.5029999999999998e+01, // C2H
  2.6037999999999997e+01, // C2H2
  2.7045999999999999e+01, // C2H3
  2.8053999999999998e+01, // C2H4
  2.9061999999999998e+01, // C2H5
  3.0070000000000000e+01, // C2H6
  1.2010999999999999e+01, // C(s)
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_imw(amrex::Real* imw_new)
{
  imw_new[0] = 4.9603174603174605e-01;  // H2
  imw_new[1] = 9.9206349206349209e-01;  // H
  imw_new[2] = 7.1291081485706148e-02;  // CH2
  imw_new[3] = 6.6511473229132020e-02;  // CH3
  imw_new[4] = 6.2332481456086768e-02;  // CH4
  imw_new[5] = 3.9952057530962849e-02;  // C2H
  imw_new[6] = 3.8405407481373381e-02;  // C2H2
  imw_new[7] = 3.6974044220956889e-02;  // C2H3
  imw_new[8] = 3.5645540742853074e-02;  // C2H4
  imw_new[9] = 3.4409194136673323e-02;  // C2H5
  imw_new[10] = 3.3255736614566010e-02; // C2H6
  imw_new[11] = 8.3257014403463503e-02; // C(s)
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
  mw_new[0] = 2.0160000000000000e+00;  // H2
  mw_new[1] = 1.0080000000000000e+00;  // H
  mw_new[2] = 1.4026999999999999e+01;  // CH2
  mw_new[3] = 1.5035000000000000e+01;  // CH3
  mw_new[4] = 1.6042999999999999e+01;  // CH4
  mw_new[5] = 2.5029999999999998e+01;  // C2H
  mw_new[6] = 2.6037999999999997e+01;  // C2H2
  mw_new[7] = 2.7045999999999999e+01;  // C2H3
  mw_new[8] = 2.8053999999999998e+01;  // C2H4
  mw_new[9] = 2.9061999999999998e+01;  // C2H5
  mw_new[10] = 3.0070000000000000e+01; // C2H6
  mw_new[11] = 1.2010999999999999e+01; // C(s)
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = +1.34433112e+00 + 7.98052075e-03 * T - 1.94781510e-05 * T2 +
                 2.01572094e-08 * T3 - 7.37611761e-12 * T4;
    // species 1: H
    species[1] = +1.50000000e+00 + 7.05332819e-13 * T - 1.99591964e-15 * T2 +
                 2.30081632e-18 * T3 - 9.27732332e-22 * T4;
    // species 2: CH2
    species[2] = +2.76267867e+00 + 9.68872143e-04 * T + 2.79489841e-06 * T2 -
                 3.85091153e-09 * T3 + 1.68741719e-12 * T4;
    // species 3: CH3
    species[3] = +2.67359040e+00 + 2.01095175e-03 * T + 5.73021856e-06 * T2 -
                 6.87117425e-09 * T3 + 2.54385734e-12 * T4;
    // species 4: CH4
    species[4] = +4.14987613e+00 - 1.36709788e-02 * T + 4.91800599e-05 * T2 -
                 4.84743026e-08 * T3 + 1.66693956e-11 * T4;
    // species 5: C2H
    species[5] = +1.88965733e+00 + 1.34099611e-02 * T - 2.84769501e-05 * T2 +
                 2.94791045e-08 * T3 - 1.09331511e-11 * T4;
    // species 6: C2H2
    species[6] = -1.91318906e-01 + 2.33615629e-02 * T - 3.55171815e-05 * T2 +
                 2.80152437e-08 * T3 - 8.50072974e-12 * T4;
    // species 7: C2H3
    species[7] = +2.21246645e+00 + 1.51479162e-03 * T + 2.59209412e-05 * T2 -
                 3.57657847e-08 * T3 + 1.47150873e-11 * T4;
    // species 8: C2H4
    species[8] = +2.95920148e+00 - 7.57052247e-03 * T + 5.70990292e-05 * T2 -
                 6.91588753e-08 * T3 + 2.69884373e-11 * T4;
    // species 9: C2H5
    species[9] = +3.30646568e+00 - 4.18658892e-03 * T + 4.97142807e-05 * T2 -
                 5.99126606e-08 * T3 + 2.30509004e-11 * T4;
    // species 10: C2H6
    species[10] = +3.29142492e+00 - 5.50154270e-03 * T + 5.99438288e-05 * T2 -
                  7.08466285e-08 * T3 + 2.68685771e-11 * T4;
    // species 11: C(s)
    species[11] = -1.31087207e+00 + 4.40353686e-03 * T + 1.90394118e-06 * T2 -
                  6.38546966e-09 * T3 + 2.98964248e-12 * T4;
  } else {
    // species 0: H2
    species[0] = +2.33727920e+00 - 4.94024731e-05 * T + 4.99456778e-07 * T2 -
                 1.79566394e-10 * T3 + 2.00255376e-14 * T4;
    // species 1: H
    species[1] = +1.50000001e+00 - 2.30842973e-11 * T + 1.61561948e-14 * T2 -
                 4.73515235e-18 * T3 + 4.98197357e-22 * T4;
    // species 2: CH2
    species[2] = +1.87410113e+00 + 3.65639292e-03 * T - 1.40894597e-06 * T2 +
                 2.60179549e-10 * T3 - 1.87727567e-14 * T4;
    // species 3: CH3
    species[3] = +1.28571772e+00 + 7.23990037e-03 * T - 2.98714348e-06 * T2 +
                 5.95684644e-10 * T3 - 4.67154394e-14 * T4;
    // species 4: CH4
    species[4] = -9.25148505e-01 + 1.33909467e-02 * T - 5.73285809e-06 * T2 +
                 1.22292535e-09 * T3 - 1.01815230e-13 * T4;
    // species 5: C2H
    species[5] = +2.16780652e+00 + 4.75221902e-03 * T - 1.83787077e-06 * T2 +
                 3.04190252e-10 * T3 - 1.77232770e-14 * T4;
    // species 6: C2H2
    species[6] = +3.14756964e+00 + 5.96166664e-03 * T - 2.37294852e-06 * T2 +
                 4.67412171e-10 * T3 - 3.61235213e-14 * T4;
    // species 7: C2H3
    species[7] = +2.01672400e+00 + 1.03302292e-02 * T - 4.68082349e-06 * T2 +
                 1.01763288e-09 * T3 - 8.62607041e-14 * T4;
    // species 8: C2H4
    species[8] = +1.03611116e+00 + 1.46454151e-02 * T - 6.71077915e-06 * T2 +
                 1.47222923e-09 * T3 - 1.25706061e-13 * T4;
    // species 9: C2H5
    species[9] = +9.54656420e-01 + 1.73972722e-02 * T - 7.98206668e-06 * T2 +
                 1.75217689e-09 * T3 - 1.49641576e-13 * T4;
    // species 10: C2H6
    species[10] = +7.18815000e-02 + 2.16852677e-02 * T - 1.00256067e-05 * T2 +
                  2.21412001e-09 * T3 - 1.90002890e-13 * T4;
    // species 11: C(s)
    species[11] = +4.55718290e-01 + 1.71702216e-03 * T - 6.97562786e-07 * T2 +
                  1.35277032e-10 * T3 - 9.67590652e-15 * T4;
  }
}

// compute Cp/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cp_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = +2.34433112e+00 + 7.98052075e-03 * T - 1.94781510e-05 * T2 +
                 2.01572094e-08 * T3 - 7.37611761e-12 * T4;
    // species 1: H
    species[1] = +2.50000000e+00 + 7.05332819e-13 * T - 1.99591964e-15 * T2 +
                 2.30081632e-18 * T3 - 9.27732332e-22 * T4;
    // species 2: CH2
    species[2] = +3.76267867e+00 + 9.68872143e-04 * T + 2.79489841e-06 * T2 -
                 3.85091153e-09 * T3 + 1.68741719e-12 * T4;
    // species 3: CH3
    species[3] = +3.67359040e+00 + 2.01095175e-03 * T + 5.73021856e-06 * T2 -
                 6.87117425e-09 * T3 + 2.54385734e-12 * T4;
    // species 4: CH4
    species[4] = +5.14987613e+00 - 1.36709788e-02 * T + 4.91800599e-05 * T2 -
                 4.84743026e-08 * T3 + 1.66693956e-11 * T4;
    // species 5: C2H
    species[5] = +2.88965733e+00 + 1.34099611e-02 * T - 2.84769501e-05 * T2 +
                 2.94791045e-08 * T3 - 1.09331511e-11 * T4;
    // species 6: C2H2
    species[6] = +8.08681094e-01 + 2.33615629e-02 * T - 3.55171815e-05 * T2 +
                 2.80152437e-08 * T3 - 8.50072974e-12 * T4;
    // species 7: C2H3
    species[7] = +3.21246645e+00 + 1.51479162e-03 * T + 2.59209412e-05 * T2 -
                 3.57657847e-08 * T3 + 1.47150873e-11 * T4;
    // species 8: C2H4
    species[8] = +3.95920148e+00 - 7.57052247e-03 * T + 5.70990292e-05 * T2 -
                 6.91588753e-08 * T3 + 2.69884373e-11 * T4;
    // species 9: C2H5
    species[9] = +4.30646568e+00 - 4.18658892e-03 * T + 4.97142807e-05 * T2 -
                 5.99126606e-08 * T3 + 2.30509004e-11 * T4;
    // species 10: C2H6
    species[10] = +4.29142492e+00 - 5.50154270e-03 * T + 5.99438288e-05 * T2 -
                  7.08466285e-08 * T3 + 2.68685771e-11 * T4;
    // species 11: C(s)
    species[11] = -3.10872072e-01 + 4.40353686e-03 * T + 1.90394118e-06 * T2 -
                  6.38546966e-09 * T3 + 2.98964248e-12 * T4;
  } else {
    // species 0: H2
    species[0] = +3.33727920e+00 - 4.94024731e-05 * T + 4.99456778e-07 * T2 -
                 1.79566394e-10 * T3 + 2.00255376e-14 * T4;
    // species 1: H
    species[1] = +2.50000001e+00 - 2.30842973e-11 * T + 1.61561948e-14 * T2 -
                 4.73515235e-18 * T3 + 4.98197357e-22 * T4;
    // species 2: CH2
    species[2] = +2.87410113e+00 + 3.65639292e-03 * T - 1.40894597e-06 * T2 +
                 2.60179549e-10 * T3 - 1.87727567e-14 * T4;
    // species 3: CH3
    species[3] = +2.28571772e+00 + 7.23990037e-03 * T - 2.98714348e-06 * T2 +
                 5.95684644e-10 * T3 - 4.67154394e-14 * T4;
    // species 4: CH4
    species[4] = +7.48514950e-02 + 1.33909467e-02 * T - 5.73285809e-06 * T2 +
                 1.22292535e-09 * T3 - 1.01815230e-13 * T4;
    // species 5: C2H
    species[5] = +3.16780652e+00 + 4.75221902e-03 * T - 1.83787077e-06 * T2 +
                 3.04190252e-10 * T3 - 1.77232770e-14 * T4;
    // species 6: C2H2
    species[6] = +4.14756964e+00 + 5.96166664e-03 * T - 2.37294852e-06 * T2 +
                 4.67412171e-10 * T3 - 3.61235213e-14 * T4;
    // species 7: C2H3
    species[7] = +3.01672400e+00 + 1.03302292e-02 * T - 4.68082349e-06 * T2 +
                 1.01763288e-09 * T3 - 8.62607041e-14 * T4;
    // species 8: C2H4
    species[8] = +2.03611116e+00 + 1.46454151e-02 * T - 6.71077915e-06 * T2 +
                 1.47222923e-09 * T3 - 1.25706061e-13 * T4;
    // species 9: C2H5
    species[9] = +1.95465642e+00 + 1.73972722e-02 * T - 7.98206668e-06 * T2 +
                 1.75217689e-09 * T3 - 1.49641576e-13 * T4;
    // species 10: C2H6
    species[10] = +1.07188150e+00 + 2.16852677e-02 * T - 1.00256067e-05 * T2 +
                  2.21412001e-09 * T3 - 1.90002890e-13 * T4;
    // species 11: C(s)
    species[11] = +1.45571829e+00 + 1.71702216e-03 * T - 6.97562786e-07 * T2 +
                  1.35277032e-10 * T3 - 9.67590652e-15 * T4;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = -9.179351730000000e+02 * invT + 1.661320882000000e+00 -
                 2.344331120000000e+00 * logT - 3.990260375000000e-03 * T +
                 3.246358500000000e-06 * T2 - 1.679767450000000e-09 * T3 +
                 3.688058805000000e-13 * T4;
    // species 1: H
    species[1] = +2.547365990000000e+04 * invT + 2.946682853000000e+00 -
                 2.500000000000000e+00 * logT - 3.526664095000000e-13 * T +
                 3.326532733333333e-16 * T2 - 1.917346933333333e-19 * T3 +
                 4.638661660000000e-23 * T4;
    // species 2: CH2
    species[2] = +4.600404010000000e+04 * invT + 2.200146820000000e+00 -
                 3.762678670000000e+00 * logT - 4.844360715000000e-04 * T -
                 4.658164016666667e-07 * T2 + 3.209092941666667e-10 * T3 -
                 8.437085950000000e-14 * T4;
    // species 3: CH3
    species[3] = +1.644499880000000e+04 * invT + 2.069026070000000e+00 -
                 3.673590400000000e+00 * logT - 1.005475875000000e-03 * T -
                 9.550364266666668e-07 * T2 + 5.725978541666666e-10 * T3 -
                 1.271928670000000e-13 * T4;
    // species 4: CH4
    species[4] = -1.024664760000000e+04 * invT + 9.791179889999999e+00 -
                 5.149876130000000e+00 * logT + 6.835489400000000e-03 * T -
                 8.196676650000000e-06 * T2 + 4.039525216666667e-09 * T3 -
                 8.334697800000000e-13 * T4;
    // species 5: C2H
    species[5] = +6.683939320000001e+04 * invT - 3.333307050000000e+00 -
                 2.889657330000000e+00 * logT - 6.704980550000000e-03 * T +
                 4.746158350000000e-06 * T2 - 2.456592041666667e-09 * T3 +
                 5.466575550000000e-13 * T4;
    // species 6: C2H2
    species[6] = +2.642898070000000e+04 * invT - 1.313102400600000e+01 -
                 8.086810940000000e-01 * logT - 1.168078145000000e-02 * T +
                 5.919530250000000e-06 * T2 - 2.334603641666667e-09 * T3 +
                 4.250364870000000e-13 * T4;
    // species 7: C2H3
    species[7] = +3.485984680000000e+04 * invT - 5.298073800000000e+00 -
                 3.212466450000000e+00 * logT - 7.573958100000000e-04 * T -
                 4.320156866666666e-06 * T2 + 2.980482058333333e-09 * T3 -
                 7.357543650000000e-13 * T4;
    // species 8: C2H4
    species[8] = +5.089775930000000e+03 * invT - 1.381294799999999e-01 -
                 3.959201480000000e+00 * logT + 3.785261235000000e-03 * T -
                 9.516504866666667e-06 * T2 + 5.763239608333333e-09 * T3 -
                 1.349421865000000e-12 * T4;
    // species 9: C2H5
    species[9] = +1.284162650000000e+04 * invT - 4.007435600000004e-01 -
                 4.306465680000000e+00 * logT + 2.093294460000000e-03 * T -
                 8.285713450000000e-06 * T2 + 4.992721716666666e-09 * T3 -
                 1.152545020000000e-12 * T4;
    // species 10: C2H6
    species[10] = -1.152220550000000e+04 * invT + 1.624601760000000e+00 -
                  4.291424920000000e+00 * logT + 2.750771350000000e-03 * T -
                  9.990638133333334e-06 * T2 + 5.903885708333334e-09 * T3 -
                  1.343428855000000e-12 * T4;
    // species 11: C(s)
    species[11] = -1.086507940000000e+02 * invT - 1.424701602000000e+00 +
                  3.108720720000000e-01 * logT - 2.201768430000000e-03 * T -
                  3.173235300000000e-07 * T2 + 5.321224716666666e-10 * T3 -
                  1.494821240000000e-13 * T4;
  } else {
    // species 0: H2
    species[0] = -9.501589220000000e+02 * invT + 6.542302510000000e+00 -
                 3.337279200000000e+00 * logT + 2.470123655000000e-05 * T -
                 8.324279633333333e-08 * T2 + 1.496386616666667e-11 * T3 -
                 1.001276880000000e-15 * T4;
    // species 1: H
    species[1] = +2.547365990000000e+04 * invT + 2.946682924000000e+00 -
                 2.500000010000000e+00 * logT + 1.154214865000000e-11 * T -
                 2.692699133333334e-15 * T2 + 3.945960291666667e-19 * T3 -
                 2.490986785000000e-23 * T4;
    // species 2: CH2
    species[2] = +4.626360400000000e+04 * invT - 3.297092110000000e+00 -
                 2.874101130000000e+00 * logT - 1.828196460000000e-03 * T +
                 2.348243283333333e-07 * T2 - 2.168162908333333e-11 * T3 +
                 9.386378350000000e-16 * T4;
    // species 3: CH3
    species[3] = +1.677558430000000e+04 * invT - 6.194354070000000e+00 -
                 2.285717720000000e+00 * logT - 3.619950185000000e-03 * T +
                 4.978572466666667e-07 * T2 - 4.964038700000000e-11 * T3 +
                 2.335771970000000e-15 * T4;
    // species 4: CH4
    species[4] = -9.468344590000001e+03 * invT - 1.836246650500000e+01 -
                 7.485149500000000e-02 * logT - 6.695473350000000e-03 * T +
                 9.554763483333333e-07 * T2 - 1.019104458333333e-10 * T3 +
                 5.090761500000000e-15 * T4;
    // species 5: C2H
    species[5] = +6.712106500000000e+04 * invT - 3.468088230000000e+00 -
                 3.167806520000000e+00 * logT - 2.376109510000000e-03 * T +
                 3.063117950000000e-07 * T2 - 2.534918766666666e-11 * T3 +
                 8.861638500000000e-16 * T4;
    // species 6: C2H2
    species[6] = +2.593599920000000e+04 * invT + 5.377850850000001e+00 -
                 4.147569640000000e+00 * logT - 2.980833320000000e-03 * T +
                 3.954914200000000e-07 * T2 - 3.895101425000000e-11 * T3 +
                 1.806176065000000e-15 * T4;
    // species 7: C2H3
    species[7] = +3.461287390000000e+04 * invT - 4.770599780000000e+00 -
                 3.016724000000000e+00 * logT - 5.165114600000000e-03 * T +
                 7.801372483333333e-07 * T2 - 8.480274000000000e-11 * T3 +
                 4.313035205000000e-15 * T4;
    // species 8: C2H4
    species[8] = +4.939886140000000e+03 * invT - 8.269258140000002e+00 -
                 2.036111160000000e+00 * logT - 7.322707550000000e-03 * T +
                 1.118463191666667e-06 * T2 - 1.226857691666667e-10 * T3 +
                 6.285303050000000e-15 * T4;
    // species 9: C2H5
    species[9] = +1.285752000000000e+04 * invT - 1.150777788000000e+01 -
                 1.954656420000000e+00 * logT - 8.698636100000001e-03 * T +
                 1.330344446666667e-06 * T2 - 1.460147408333333e-10 * T3 +
                 7.482078800000000e-15 * T4;
    // species 10: C2H6
    species[10] = -1.142639320000000e+04 * invT - 1.404372920000000e+01 -
                  1.071881500000000e+00 * logT - 1.084263385000000e-02 * T +
                  1.670934450000000e-06 * T2 - 1.845100008333333e-10 * T3 +
                  9.500144500000000e-15 * T4;
    // species 11: C(s)
    species[11] = -6.951388140000000e+02 * invT + 9.981548620000000e+00 -
                  1.455718290000000e+00 * logT - 8.585110800000000e-04 * T +
                  1.162604643333333e-07 * T2 - 1.127308600000000e-11 * T3 +
                  4.837953260000000e-16 * T4;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = -9.17935173e+02 * invT + 6.61320882e-01 -
                 2.34433112e+00 * logT - 3.99026037e-03 * T +
                 3.24635850e-06 * T2 - 1.67976745e-09 * T3 +
                 3.68805881e-13 * T4;
    // species 1: H
    species[1] = +2.54736599e+04 * invT + 1.94668285e+00 -
                 2.50000000e+00 * logT - 3.52666409e-13 * T +
                 3.32653273e-16 * T2 - 1.91734693e-19 * T3 +
                 4.63866166e-23 * T4;
    // species 2: CH2
    species[2] = +4.60040401e+04 * invT + 1.20014682e+00 -
                 3.76267867e+00 * logT - 4.84436072e-04 * T -
                 4.65816402e-07 * T2 + 3.20909294e-10 * T3 -
                 8.43708595e-14 * T4;
    // species 3: CH3
    species[3] = +1.64449988e+04 * invT + 1.06902607e+00 -
                 3.67359040e+00 * logT - 1.00547588e-03 * T -
                 9.55036427e-07 * T2 + 5.72597854e-10 * T3 -
                 1.27192867e-13 * T4;
    // species 4: CH4
    species[4] = -1.02466476e+04 * invT + 8.79117989e+00 -
                 5.14987613e+00 * logT + 6.83548940e-03 * T -
                 8.19667665e-06 * T2 + 4.03952522e-09 * T3 -
                 8.33469780e-13 * T4;
    // species 5: C2H
    species[5] = +6.68393932e+04 * invT - 4.33330705e+00 -
                 2.88965733e+00 * logT - 6.70498055e-03 * T +
                 4.74615835e-06 * T2 - 2.45659204e-09 * T3 +
                 5.46657555e-13 * T4;
    // species 6: C2H2
    species[6] = +2.64289807e+04 * invT - 1.41310240e+01 -
                 8.08681094e-01 * logT - 1.16807815e-02 * T +
                 5.91953025e-06 * T2 - 2.33460364e-09 * T3 +
                 4.25036487e-13 * T4;
    // species 7: C2H3
    species[7] = +3.48598468e+04 * invT - 6.29807380e+00 -
                 3.21246645e+00 * logT - 7.57395810e-04 * T -
                 4.32015687e-06 * T2 + 2.98048206e-09 * T3 -
                 7.35754365e-13 * T4;
    // species 8: C2H4
    species[8] = +5.08977593e+03 * invT - 1.13812948e+00 -
                 3.95920148e+00 * logT + 3.78526124e-03 * T -
                 9.51650487e-06 * T2 + 5.76323961e-09 * T3 -
                 1.34942187e-12 * T4;
    // species 9: C2H5
    species[9] = +1.28416265e+04 * invT - 1.40074356e+00 -
                 4.30646568e+00 * logT + 2.09329446e-03 * T -
                 8.28571345e-06 * T2 + 4.99272172e-09 * T3 -
                 1.15254502e-12 * T4;
    // species 10: C2H6
    species[10] = -1.15222055e+04 * invT + 6.24601760e-01 -
                  4.29142492e+00 * logT + 2.75077135e-03 * T -
                  9.99063813e-06 * T2 + 5.90388571e-09 * T3 -
                  1.34342886e-12 * T4;
    // species 11: C(s)
    species[11] = -1.08650794e+02 * invT - 2.42470160e+00 +
                  3.10872072e-01 * logT - 2.20176843e-03 * T -
                  3.17323530e-07 * T2 + 5.32122472e-10 * T3 -
                  1.49482124e-13 * T4;
  } else {
    // species 0: H2
    species[0] = -9.50158922e+02 * invT + 5.54230251e+00 -
                 3.33727920e+00 * logT + 2.47012365e-05 * T -
                 8.32427963e-08 * T2 + 1.49638662e-11 * T3 -
                 1.00127688e-15 * T4;
    // species 1: H
    species[1] = +2.54736599e+04 * invT + 1.94668292e+00 -
                 2.50000001e+00 * logT + 1.15421486e-11 * T -
                 2.69269913e-15 * T2 + 3.94596029e-19 * T3 -
                 2.49098679e-23 * T4;
    // species 2: CH2
    species[2] = +4.62636040e+04 * invT - 4.29709211e+00 -
                 2.87410113e+00 * logT - 1.82819646e-03 * T +
                 2.34824328e-07 * T2 - 2.16816291e-11 * T3 +
                 9.38637835e-16 * T4;
    // species 3: CH3
    species[3] = +1.67755843e+04 * invT - 7.19435407e+00 -
                 2.28571772e+00 * logT - 3.61995018e-03 * T +
                 4.97857247e-07 * T2 - 4.96403870e-11 * T3 +
                 2.33577197e-15 * T4;
    // species 4: CH4
    species[4] = -9.46834459e+03 * invT - 1.93624665e+01 -
                 7.48514950e-02 * logT - 6.69547335e-03 * T +
                 9.55476348e-07 * T2 - 1.01910446e-10 * T3 +
                 5.09076150e-15 * T4;
    // species 5: C2H
    species[5] = +6.71210650e+04 * invT - 4.46808823e+00 -
                 3.16780652e+00 * logT - 2.37610951e-03 * T +
                 3.06311795e-07 * T2 - 2.53491877e-11 * T3 +
                 8.86163850e-16 * T4;
    // species 6: C2H2
    species[6] = +2.59359992e+04 * invT + 4.37785085e+00 -
                 4.14756964e+00 * logT - 2.98083332e-03 * T +
                 3.95491420e-07 * T2 - 3.89510143e-11 * T3 +
                 1.80617607e-15 * T4;
    // species 7: C2H3
    species[7] = +3.46128739e+04 * invT - 5.77059978e+00 -
                 3.01672400e+00 * logT - 5.16511460e-03 * T +
                 7.80137248e-07 * T2 - 8.48027400e-11 * T3 +
                 4.31303520e-15 * T4;
    // species 8: C2H4
    species[8] = +4.93988614e+03 * invT - 9.26925814e+00 -
                 2.03611116e+00 * logT - 7.32270755e-03 * T +
                 1.11846319e-06 * T2 - 1.22685769e-10 * T3 +
                 6.28530305e-15 * T4;
    // species 9: C2H5
    species[9] = +1.28575200e+04 * invT - 1.25077779e+01 -
                 1.95465642e+00 * logT - 8.69863610e-03 * T +
                 1.33034445e-06 * T2 - 1.46014741e-10 * T3 +
                 7.48207880e-15 * T4;
    // species 10: C2H6
    species[10] = -1.14263932e+04 * invT - 1.50437292e+01 -
                  1.07188150e+00 * logT - 1.08426339e-02 * T +
                  1.67093445e-06 * T2 - 1.84510001e-10 * T3 +
                  9.50014450e-15 * T4;
    // species 11: C(s)
    species[11] = -6.95138814e+02 * invT + 8.98154862e+00 -
                  1.45571829e+00 * logT - 8.58511080e-04 * T +
                  1.16260464e-07 * T2 - 1.12730860e-11 * T3 +
                  4.83795326e-16 * T4;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = +1.34433112e+00 + 3.99026037e-03 * T - 6.49271700e-06 * T2 +
                 5.03930235e-09 * T3 - 1.47522352e-12 * T4 -
                 9.17935173e+02 * invT;
    // species 1: H
    species[1] = +1.50000000e+00 + 3.52666409e-13 * T - 6.65306547e-16 * T2 +
                 5.75204080e-19 * T3 - 1.85546466e-22 * T4 +
                 2.54736599e+04 * invT;
    // species 2: CH2
    species[2] = +2.76267867e+00 + 4.84436072e-04 * T + 9.31632803e-07 * T2 -
                 9.62727883e-10 * T3 + 3.37483438e-13 * T4 +
                 4.60040401e+04 * invT;
    // species 3: CH3
    species[3] = +2.67359040e+00 + 1.00547588e-03 * T + 1.91007285e-06 * T2 -
                 1.71779356e-09 * T3 + 5.08771468e-13 * T4 +
                 1.64449988e+04 * invT;
    // species 4: CH4
    species[4] = +4.14987613e+00 - 6.83548940e-03 * T + 1.63933533e-05 * T2 -
                 1.21185757e-08 * T3 + 3.33387912e-12 * T4 -
                 1.02466476e+04 * invT;
    // species 5: C2H
    species[5] = +1.88965733e+00 + 6.70498055e-03 * T - 9.49231670e-06 * T2 +
                 7.36977613e-09 * T3 - 2.18663022e-12 * T4 +
                 6.68393932e+04 * invT;
    // species 6: C2H2
    species[6] = -1.91318906e-01 + 1.16807815e-02 * T - 1.18390605e-05 * T2 +
                 7.00381092e-09 * T3 - 1.70014595e-12 * T4 +
                 2.64289807e+04 * invT;
    // species 7: C2H3
    species[7] = +2.21246645e+00 + 7.57395810e-04 * T + 8.64031373e-06 * T2 -
                 8.94144617e-09 * T3 + 2.94301746e-12 * T4 +
                 3.48598468e+04 * invT;
    // species 8: C2H4
    species[8] = +2.95920148e+00 - 3.78526124e-03 * T + 1.90330097e-05 * T2 -
                 1.72897188e-08 * T3 + 5.39768746e-12 * T4 +
                 5.08977593e+03 * invT;
    // species 9: C2H5
    species[9] = +3.30646568e+00 - 2.09329446e-03 * T + 1.65714269e-05 * T2 -
                 1.49781651e-08 * T3 + 4.61018008e-12 * T4 +
                 1.28416265e+04 * invT;
    // species 10: C2H6
    species[10] = +3.29142492e+00 - 2.75077135e-03 * T + 1.99812763e-05 * T2 -
                  1.77116571e-08 * T3 + 5.37371542e-12 * T4 -
                  1.15222055e+04 * invT;
    // species 11: C(s)
    species[11] = -1.31087207e+00 + 2.20176843e-03 * T + 6.34647060e-07 * T2 -
                  1.59636741e-09 * T3 + 5.97928496e-13 * T4 -
                  1.08650794e+02 * invT;
  } else {
    // species 0: H2
    species[0] = +2.33727920e+00 - 2.47012365e-05 * T + 1.66485593e-07 * T2 -
                 4.48915985e-11 * T3 + 4.00510752e-15 * T4 -
                 9.50158922e+02 * invT;
    // species 1: H
    species[1] = +1.50000001e+00 - 1.15421486e-11 * T + 5.38539827e-15 * T2 -
                 1.18378809e-18 * T3 + 9.96394714e-23 * T4 +
                 2.54736599e+04 * invT;
    // species 2: CH2
    species[2] = +1.87410113e+00 + 1.82819646e-03 * T - 4.69648657e-07 * T2 +
                 6.50448872e-11 * T3 - 3.75455134e-15 * T4 +
                 4.62636040e+04 * invT;
    // species 3: CH3
    species[3] = +1.28571772e+00 + 3.61995018e-03 * T - 9.95714493e-07 * T2 +
                 1.48921161e-10 * T3 - 9.34308788e-15 * T4 +
                 1.67755843e+04 * invT;
    // species 4: CH4
    species[4] = -9.25148505e-01 + 6.69547335e-03 * T - 1.91095270e-06 * T2 +
                 3.05731338e-10 * T3 - 2.03630460e-14 * T4 -
                 9.46834459e+03 * invT;
    // species 5: C2H
    species[5] = +2.16780652e+00 + 2.37610951e-03 * T - 6.12623590e-07 * T2 +
                 7.60475630e-11 * T3 - 3.54465540e-15 * T4 +
                 6.71210650e+04 * invT;
    // species 6: C2H2
    species[6] = +3.14756964e+00 + 2.98083332e-03 * T - 7.90982840e-07 * T2 +
                 1.16853043e-10 * T3 - 7.22470426e-15 * T4 +
                 2.59359992e+04 * invT;
    // species 7: C2H3
    species[7] = +2.01672400e+00 + 5.16511460e-03 * T - 1.56027450e-06 * T2 +
                 2.54408220e-10 * T3 - 1.72521408e-14 * T4 +
                 3.46128739e+04 * invT;
    // species 8: C2H4
    species[8] = +1.03611116e+00 + 7.32270755e-03 * T - 2.23692638e-06 * T2 +
                 3.68057308e-10 * T3 - 2.51412122e-14 * T4 +
                 4.93988614e+03 * invT;
    // species 9: C2H5
    species[9] = +9.54656420e-01 + 8.69863610e-03 * T - 2.66068889e-06 * T2 +
                 4.38044223e-10 * T3 - 2.99283152e-14 * T4 +
                 1.28575200e+04 * invT;
    // species 10: C2H6
    species[10] = +7.18815000e-02 + 1.08426339e-02 * T - 3.34186890e-06 * T2 +
                  5.53530003e-10 * T3 - 3.80005780e-14 * T4 -
                  1.14263932e+04 * invT;
    // species 11: C(s)
    species[11] = +4.55718290e-01 + 8.58511080e-04 * T - 2.32520929e-07 * T2 +
                  3.38192580e-11 * T3 - 1.93518130e-15 * T4 -
                  6.95138814e+02 * invT;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = +2.34433112e+00 + 3.99026037e-03 * T - 6.49271700e-06 * T2 +
                 5.03930235e-09 * T3 - 1.47522352e-12 * T4 -
                 9.17935173e+02 * invT;
    // species 1: H
    species[1] = +2.50000000e+00 + 3.52666409e-13 * T - 6.65306547e-16 * T2 +
                 5.75204080e-19 * T3 - 1.85546466e-22 * T4 +
                 2.54736599e+04 * invT;
    // species 2: CH2
    species[2] = +3.76267867e+00 + 4.84436072e-04 * T + 9.31632803e-07 * T2 -
                 9.62727883e-10 * T3 + 3.37483438e-13 * T4 +
                 4.60040401e+04 * invT;
    // species 3: CH3
    species[3] = +3.67359040e+00 + 1.00547588e-03 * T + 1.91007285e-06 * T2 -
                 1.71779356e-09 * T3 + 5.08771468e-13 * T4 +
                 1.64449988e+04 * invT;
    // species 4: CH4
    species[4] = +5.14987613e+00 - 6.83548940e-03 * T + 1.63933533e-05 * T2 -
                 1.21185757e-08 * T3 + 3.33387912e-12 * T4 -
                 1.02466476e+04 * invT;
    // species 5: C2H
    species[5] = +2.88965733e+00 + 6.70498055e-03 * T - 9.49231670e-06 * T2 +
                 7.36977613e-09 * T3 - 2.18663022e-12 * T4 +
                 6.68393932e+04 * invT;
    // species 6: C2H2
    species[6] = +8.08681094e-01 + 1.16807815e-02 * T - 1.18390605e-05 * T2 +
                 7.00381092e-09 * T3 - 1.70014595e-12 * T4 +
                 2.64289807e+04 * invT;
    // species 7: C2H3
    species[7] = +3.21246645e+00 + 7.57395810e-04 * T + 8.64031373e-06 * T2 -
                 8.94144617e-09 * T3 + 2.94301746e-12 * T4 +
                 3.48598468e+04 * invT;
    // species 8: C2H4
    species[8] = +3.95920148e+00 - 3.78526124e-03 * T + 1.90330097e-05 * T2 -
                 1.72897188e-08 * T3 + 5.39768746e-12 * T4 +
                 5.08977593e+03 * invT;
    // species 9: C2H5
    species[9] = +4.30646568e+00 - 2.09329446e-03 * T + 1.65714269e-05 * T2 -
                 1.49781651e-08 * T3 + 4.61018008e-12 * T4 +
                 1.28416265e+04 * invT;
    // species 10: C2H6
    species[10] = +4.29142492e+00 - 2.75077135e-03 * T + 1.99812763e-05 * T2 -
                  1.77116571e-08 * T3 + 5.37371542e-12 * T4 -
                  1.15222055e+04 * invT;
    // species 11: C(s)
    species[11] = -3.10872072e-01 + 2.20176843e-03 * T + 6.34647060e-07 * T2 -
                  1.59636741e-09 * T3 + 5.97928496e-13 * T4 -
                  1.08650794e+02 * invT;
  } else {
    // species 0: H2
    species[0] = +3.33727920e+00 - 2.47012365e-05 * T + 1.66485593e-07 * T2 -
                 4.48915985e-11 * T3 + 4.00510752e-15 * T4 -
                 9.50158922e+02 * invT;
    // species 1: H
    species[1] = +2.50000001e+00 - 1.15421486e-11 * T + 5.38539827e-15 * T2 -
                 1.18378809e-18 * T3 + 9.96394714e-23 * T4 +
                 2.54736599e+04 * invT;
    // species 2: CH2
    species[2] = +2.87410113e+00 + 1.82819646e-03 * T - 4.69648657e-07 * T2 +
                 6.50448872e-11 * T3 - 3.75455134e-15 * T4 +
                 4.62636040e+04 * invT;
    // species 3: CH3
    species[3] = +2.28571772e+00 + 3.61995018e-03 * T - 9.95714493e-07 * T2 +
                 1.48921161e-10 * T3 - 9.34308788e-15 * T4 +
                 1.67755843e+04 * invT;
    // species 4: CH4
    species[4] = +7.48514950e-02 + 6.69547335e-03 * T - 1.91095270e-06 * T2 +
                 3.05731338e-10 * T3 - 2.03630460e-14 * T4 -
                 9.46834459e+03 * invT;
    // species 5: C2H
    species[5] = +3.16780652e+00 + 2.37610951e-03 * T - 6.12623590e-07 * T2 +
                 7.60475630e-11 * T3 - 3.54465540e-15 * T4 +
                 6.71210650e+04 * invT;
    // species 6: C2H2
    species[6] = +4.14756964e+00 + 2.98083332e-03 * T - 7.90982840e-07 * T2 +
                 1.16853043e-10 * T3 - 7.22470426e-15 * T4 +
                 2.59359992e+04 * invT;
    // species 7: C2H3
    species[7] = +3.01672400e+00 + 5.16511460e-03 * T - 1.56027450e-06 * T2 +
                 2.54408220e-10 * T3 - 1.72521408e-14 * T4 +
                 3.46128739e+04 * invT;
    // species 8: C2H4
    species[8] = +2.03611116e+00 + 7.32270755e-03 * T - 2.23692638e-06 * T2 +
                 3.68057308e-10 * T3 - 2.51412122e-14 * T4 +
                 4.93988614e+03 * invT;
    // species 9: C2H5
    species[9] = +1.95465642e+00 + 8.69863610e-03 * T - 2.66068889e-06 * T2 +
                 4.38044223e-10 * T3 - 2.99283152e-14 * T4 +
                 1.28575200e+04 * invT;
    // species 10: C2H6
    species[10] = +1.07188150e+00 + 1.08426339e-02 * T - 3.34186890e-06 * T2 +
                  5.53530003e-10 * T3 - 3.80005780e-14 * T4 -
                  1.14263932e+04 * invT;
    // species 11: C(s)
    species[11] = +1.45571829e+00 + 8.58511080e-04 * T - 2.32520929e-07 * T2 +
                  3.38192580e-11 * T3 - 1.93518130e-15 * T4 -
                  6.95138814e+02 * invT;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = +2.34433112e+00 * logT + 7.98052075e-03 * T -
                 9.73907550e-06 * T2 + 6.71906980e-09 * T3 -
                 1.84402940e-12 * T4 + 6.83010238e-01;
    // species 1: H
    species[1] = +2.50000000e+00 * logT + 7.05332819e-13 * T -
                 9.97959820e-16 * T2 + 7.66938773e-19 * T3 -
                 2.31933083e-22 * T4 - 4.46682853e-01;
    // species 2: CH2
    species[2] = +3.76267867e+00 * logT + 9.68872143e-04 * T +
                 1.39744921e-06 * T2 - 1.28363718e-09 * T3 +
                 4.21854298e-13 * T4 + 1.56253185e+00;
    // species 3: CH3
    species[3] = +3.67359040e+00 * logT + 2.01095175e-03 * T +
                 2.86510928e-06 * T2 - 2.29039142e-09 * T3 +
                 6.35964335e-13 * T4 + 1.60456433e+00;
    // species 4: CH4
    species[4] = +5.14987613e+00 * logT - 1.36709788e-02 * T +
                 2.45900299e-05 * T2 - 1.61581009e-08 * T3 +
                 4.16734890e-12 * T4 - 4.64130376e+00;
    // species 5: C2H
    species[5] = +2.88965733e+00 * logT + 1.34099611e-02 * T -
                 1.42384751e-05 * T2 + 9.82636817e-09 * T3 -
                 2.73328777e-12 * T4 + 6.22296438e+00;
    // species 6: C2H2
    species[6] = +8.08681094e-01 * logT + 2.33615629e-02 * T -
                 1.77585907e-05 * T2 + 9.33841457e-09 * T3 -
                 2.12518243e-12 * T4 + 1.39397051e+01;
    // species 7: C2H3
    species[7] = +3.21246645e+00 * logT + 1.51479162e-03 * T +
                 1.29604706e-05 * T2 - 1.19219282e-08 * T3 +
                 3.67877182e-12 * T4 + 8.51054025e+00;
    // species 8: C2H4
    species[8] = +3.95920148e+00 * logT - 7.57052247e-03 * T +
                 2.85495146e-05 * T2 - 2.30529584e-08 * T3 +
                 6.74710933e-12 * T4 + 4.09733096e+00;
    // species 9: C2H5
    species[9] = +4.30646568e+00 * logT - 4.18658892e-03 * T +
                 2.48571403e-05 * T2 - 1.99708869e-08 * T3 +
                 5.76272510e-12 * T4 + 4.70720924e+00;
    // species 10: C2H6
    species[10] = +4.29142492e+00 * logT - 5.50154270e-03 * T +
                  2.99719144e-05 * T2 - 2.36155428e-08 * T3 +
                  6.71714427e-12 * T4 + 2.66682316e+00;
    // species 11: C(s)
    species[11] = -3.10872072e-01 * logT + 4.40353686e-03 * T +
                  9.51970590e-07 * T2 - 2.12848989e-09 * T3 +
                  7.47410620e-13 * T4 + 1.11382953e+00;
  } else {
    // species 0: H2
    species[0] = +3.33727920e+00 * logT - 4.94024731e-05 * T +
                 2.49728389e-07 * T2 - 5.98554647e-11 * T3 +
                 5.00638440e-15 * T4 - 3.20502331e+00;
    // species 1: H
    species[1] = +2.50000001e+00 * logT - 2.30842973e-11 * T +
                 8.07809740e-15 * T2 - 1.57838412e-18 * T3 +
                 1.24549339e-22 * T4 - 4.46682914e-01;
    // species 2: CH2
    species[2] = +2.87410113e+00 * logT + 3.65639292e-03 * T -
                 7.04472985e-07 * T2 + 8.67265163e-11 * T3 -
                 4.69318918e-15 * T4 + 6.17119324e+00;
    // species 3: CH3
    species[3] = +2.28571772e+00 * logT + 7.23990037e-03 * T -
                 1.49357174e-06 * T2 + 1.98561548e-10 * T3 -
                 1.16788599e-14 * T4 + 8.48007179e+00;
    // species 4: CH4
    species[4] = +7.48514950e-02 * logT + 1.33909467e-02 * T -
                 2.86642905e-06 * T2 + 4.07641783e-10 * T3 -
                 2.54538075e-14 * T4 + 1.84373180e+01;
    // species 5: C2H
    species[5] = +3.16780652e+00 * logT + 4.75221902e-03 * T -
                 9.18935385e-07 * T2 + 1.01396751e-10 * T3 -
                 4.43081925e-15 * T4 + 6.63589475e+00;
    // species 6: C2H2
    species[6] = +4.14756964e+00 * logT + 5.96166664e-03 * T -
                 1.18647426e-06 * T2 + 1.55804057e-10 * T3 -
                 9.03088033e-15 * T4 - 1.23028121e+00;
    // species 7: C2H3
    species[7] = +3.01672400e+00 * logT + 1.03302292e-02 * T -
                 2.34041174e-06 * T2 + 3.39210960e-10 * T3 -
                 2.15651760e-14 * T4 + 7.78732378e+00;
    // species 8: C2H4
    species[8] = +2.03611116e+00 * logT + 1.46454151e-02 * T -
                 3.35538958e-06 * T2 + 4.90743077e-10 * T3 -
                 3.14265152e-14 * T4 + 1.03053693e+01;
    // species 9: C2H5
    species[9] = +1.95465642e+00 * logT + 1.73972722e-02 * T -
                 3.99103334e-06 * T2 + 5.84058963e-10 * T3 -
                 3.74103940e-14 * T4 + 1.34624343e+01;
    // species 10: C2H6
    species[10] = +1.07188150e+00 * logT + 2.16852677e-02 * T -
                  5.01280335e-06 * T2 + 7.38040003e-10 * T3 -
                  4.75007225e-14 * T4 + 1.51156107e+01;
    // species 11: C(s)
    species[11] = +1.45571829e+00 * logT + 1.71702216e-03 * T -
                  3.48781393e-07 * T2 + 4.50923440e-11 * T3 -
                  2.41897663e-15 * T4 - 8.52583033e+00;
  }
}

// compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
dcvpRdT(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = +7.98052075e-03 - 3.89563020e-05 * T + 6.04716282e-08 * T2 -
                 2.95044704e-11 * T3;
    // species 1: H
    species[1] = +7.05332819e-13 - 3.99183928e-15 * T + 6.90244896e-18 * T2 -
                 3.71092933e-21 * T3;
    // species 2: CH2
    species[2] = +9.68872143e-04 + 5.58979682e-06 * T - 1.15527346e-08 * T2 +
                 6.74966876e-12 * T3;
    // species 3: CH3
    species[3] = +2.01095175e-03 + 1.14604371e-05 * T - 2.06135228e-08 * T2 +
                 1.01754294e-11 * T3;
    // species 4: CH4
    species[4] = -1.36709788e-02 + 9.83601198e-05 * T - 1.45422908e-07 * T2 +
                 6.66775824e-11 * T3;
    // species 5: C2H
    species[5] = +1.34099611e-02 - 5.69539002e-05 * T + 8.84373135e-08 * T2 -
                 4.37326044e-11 * T3;
    // species 6: C2H2
    species[6] = +2.33615629e-02 - 7.10343630e-05 * T + 8.40457311e-08 * T2 -
                 3.40029190e-11 * T3;
    // species 7: C2H3
    species[7] = +1.51479162e-03 + 5.18418824e-05 * T - 1.07297354e-07 * T2 +
                 5.88603492e-11 * T3;
    // species 8: C2H4
    species[8] = -7.57052247e-03 + 1.14198058e-04 * T - 2.07476626e-07 * T2 +
                 1.07953749e-10 * T3;
    // species 9: C2H5
    species[9] = -4.18658892e-03 + 9.94285614e-05 * T - 1.79737982e-07 * T2 +
                 9.22036016e-11 * T3;
    // species 10: C2H6
    species[10] = -5.50154270e-03 + 1.19887658e-04 * T - 2.12539886e-07 * T2 +
                  1.07474308e-10 * T3;
    // species 11: C(s)
    species[11] = +4.40353686e-03 + 3.80788236e-06 * T - 1.91564090e-08 * T2 +
                  1.19585699e-11 * T3;
  } else {
    // species 0: H2
    species[0] = -4.94024731e-05 + 9.98913556e-07 * T - 5.38699182e-10 * T2 +
                 8.01021504e-14 * T3;
    // species 1: H
    species[1] = -2.30842973e-11 + 3.23123896e-14 * T - 1.42054571e-17 * T2 +
                 1.99278943e-21 * T3;
    // species 2: CH2
    species[2] = +3.65639292e-03 - 2.81789194e-06 * T + 7.80538647e-10 * T2 -
                 7.50910268e-14 * T3;
    // species 3: CH3
    species[3] = +7.23990037e-03 - 5.97428696e-06 * T + 1.78705393e-09 * T2 -
                 1.86861758e-13 * T3;
    // species 4: CH4
    species[4] = +1.33909467e-02 - 1.14657162e-05 * T + 3.66877605e-09 * T2 -
                 4.07260920e-13 * T3;
    // species 5: C2H
    species[5] = +4.75221902e-03 - 3.67574154e-06 * T + 9.12570756e-10 * T2 -
                 7.08931080e-14 * T3;
    // species 6: C2H2
    species[6] = +5.96166664e-03 - 4.74589704e-06 * T + 1.40223651e-09 * T2 -
                 1.44494085e-13 * T3;
    // species 7: C2H3
    species[7] = +1.03302292e-02 - 9.36164698e-06 * T + 3.05289864e-09 * T2 -
                 3.45042816e-13 * T3;
    // species 8: C2H4
    species[8] = +1.46454151e-02 - 1.34215583e-05 * T + 4.41668769e-09 * T2 -
                 5.02824244e-13 * T3;
    // species 9: C2H5
    species[9] = +1.73972722e-02 - 1.59641334e-05 * T + 5.25653067e-09 * T2 -
                 5.98566304e-13 * T3;
    // species 10: C2H6
    species[10] = +2.16852677e-02 - 2.00512134e-05 * T + 6.64236003e-09 * T2 -
                  7.60011560e-13 * T3;
    // species 11: C(s)
    species[11] = +1.71702216e-03 - 1.39512557e-06 * T + 4.05831096e-10 * T2 -
                  3.87036261e-14 * T3;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    result += y[0] *
              (+2.34433112e+00 + 7.98052075e-03 * T - 1.94781510e-05 * T2 +
               2.01572094e-08 * T3 - 7.37611761e-12 * T4) *
              0.4960317460317460;
    // species 1: H
    result += y[1] *
              (+2.50000000e+00 + 7.05332819e-13 * T - 1.99591964e-15 * T2 +
               2.30081632e-18 * T3 - 9.27732332e-22 * T4) *
              0.9920634920634921;
    // species 2: CH2
    result += y[2] *
              (+3.76267867e+00 + 9.68872143e-04 * T + 2.79489841e-06 * T2 -
               3.85091153e-09 * T3 + 1.68741719e-12 * T4) *
              0.0712910814857061;
    // species 3: CH3
    result += y[3] *
              (+3.67359040e+00 + 2.01095175e-03 * T + 5.73021856e-06 * T2 -
               6.87117425e-09 * T3 + 2.54385734e-12 * T4) *
              0.0665114732291320;
    // species 4: CH4
    result += y[4] *
              (+5.14987613e+00 - 1.36709788e-02 * T + 4.91800599e-05 * T2 -
               4.84743026e-08 * T3 + 1.66693956e-11 * T4) *
              0.0623324814560868;
    // species 5: C2H
    result += y[5] *
              (+2.88965733e+00 + 1.34099611e-02 * T - 2.84769501e-05 * T2 +
               2.94791045e-08 * T3 - 1.09331511e-11 * T4) *
              0.0399520575309628;
    // species 6: C2H2
    result += y[6] *
              (+8.08681094e-01 + 2.33615629e-02 * T - 3.55171815e-05 * T2 +
               2.80152437e-08 * T3 - 8.50072974e-12 * T4) *
              0.0384054074813734;
    // species 7: C2H3
    result += y[7] *
              (+3.21246645e+00 + 1.51479162e-03 * T + 2.59209412e-05 * T2 -
               3.57657847e-08 * T3 + 1.47150873e-11 * T4) *
              0.0369740442209569;
    // species 8: C2H4
    result += y[8] *
              (+3.95920148e+00 - 7.57052247e-03 * T + 5.70990292e-05 * T2 -
               6.91588753e-08 * T3 + 2.69884373e-11 * T4) *
              0.0356455407428531;
    // species 9: C2H5
    result += y[9] *
              (+4.30646568e+00 - 4.18658892e-03 * T + 4.97142807e-05 * T2 -
               5.99126606e-08 * T3 + 2.30509004e-11 * T4) *
              0.0344091941366733;
    // species 10: C2H6
    result += y[10] *
              (+4.29142492e+00 - 5.50154270e-03 * T + 5.99438288e-05 * T2 -
               7.08466285e-08 * T3 + 2.68685771e-11 * T4) *
              0.0332557366145660;
    // species 11: C(s)
    result += y[11] *
              (-3.10872072e-01 + 4.40353686e-03 * T + 1.90394118e-06 * T2 -
               6.38546966e-09 * T3 + 2.98964248e-12 * T4) *
              0.0832570144034635;
  } else {
    // species 0: H2
    result += y[0] *
              (+3.33727920e+00 - 4.94024731e-05 * T + 4.99456778e-07 * T2 -
               1.79566394e-10 * T3 + 2.00255376e-14 * T4) *
              0.4960317460317460;
    // species 1: H
    result += y[1] *
              (+2.50000001e+00 - 2.30842973e-11 * T + 1.61561948e-14 * T2 -
               4.73515235e-18 * T3 + 4.98197357e-22 * T4) *
              0.9920634920634921;
    // species 2: CH2
    result += y[2] *
              (+2.87410113e+00 + 3.65639292e-03 * T - 1.40894597e-06 * T2 +
               2.60179549e-10 * T3 - 1.87727567e-14 * T4) *
              0.0712910814857061;
    // species 3: CH3
    result += y[3] *
              (+2.28571772e+00 + 7.23990037e-03 * T - 2.98714348e-06 * T2 +
               5.95684644e-10 * T3 - 4.67154394e-14 * T4) *
              0.0665114732291320;
    // species 4: CH4
    result += y[4] *
              (+7.48514950e-02 + 1.33909467e-02 * T - 5.73285809e-06 * T2 +
               1.22292535e-09 * T3 - 1.01815230e-13 * T4) *
              0.0623324814560868;
    // species 5: C2H
    result += y[5] *
              (+3.16780652e+00 + 4.75221902e-03 * T - 1.83787077e-06 * T2 +
               3.04190252e-10 * T3 - 1.77232770e-14 * T4) *
              0.0399520575309628;
    // species 6: C2H2
    result += y[6] *
              (+4.14756964e+00 + 5.96166664e-03 * T - 2.37294852e-06 * T2 +
               4.67412171e-10 * T3 - 3.61235213e-14 * T4) *
              0.0384054074813734;
    // species 7: C2H3
    result += y[7] *
              (+3.01672400e+00 + 1.03302292e-02 * T - 4.68082349e-06 * T2 +
               1.01763288e-09 * T3 - 8.62607041e-14 * T4) *
              0.0369740442209569;
    // species 8: C2H4
    result += y[8] *
              (+2.03611116e+00 + 1.46454151e-02 * T - 6.71077915e-06 * T2 +
               1.47222923e-09 * T3 - 1.25706061e-13 * T4) *
              0.0356455407428531;
    // species 9: C2H5
    result += y[9] *
              (+1.95465642e+00 + 1.73972722e-02 * T - 7.98206668e-06 * T2 +
               1.75217689e-09 * T3 - 1.49641576e-13 * T4) *
              0.0344091941366733;
    // species 10: C2H6
    result += y[10] *
              (+1.07188150e+00 + 2.16852677e-02 * T - 1.00256067e-05 * T2 +
               2.21412001e-09 * T3 - 1.90002890e-13 * T4) *
              0.0332557366145660;
    // species 11: C(s)
    result += y[11] *
              (+1.45571829e+00 + 1.71702216e-03 * T - 6.97562786e-07 * T2 +
               1.35277032e-10 * T3 - 9.67590652e-15 * T4) *
              0.0832570144034635;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    result += y[0] *
              (+1.34433112e+00 + 7.98052075e-03 * T - 1.94781510e-05 * T2 +
               2.01572094e-08 * T3 - 7.37611761e-12 * T4) *
              0.4960317460317460;
    // species 1: H
    result += y[1] *
              (+1.50000000e+00 + 7.05332819e-13 * T - 1.99591964e-15 * T2 +
               2.30081632e-18 * T3 - 9.27732332e-22 * T4) *
              0.9920634920634921;
    // species 2: CH2
    result += y[2] *
              (+2.76267867e+00 + 9.68872143e-04 * T + 2.79489841e-06 * T2 -
               3.85091153e-09 * T3 + 1.68741719e-12 * T4) *
              0.0712910814857061;
    // species 3: CH3
    result += y[3] *
              (+2.67359040e+00 + 2.01095175e-03 * T + 5.73021856e-06 * T2 -
               6.87117425e-09 * T3 + 2.54385734e-12 * T4) *
              0.0665114732291320;
    // species 4: CH4
    result += y[4] *
              (+4.14987613e+00 - 1.36709788e-02 * T + 4.91800599e-05 * T2 -
               4.84743026e-08 * T3 + 1.66693956e-11 * T4) *
              0.0623324814560868;
    // species 5: C2H
    result += y[5] *
              (+1.88965733e+00 + 1.34099611e-02 * T - 2.84769501e-05 * T2 +
               2.94791045e-08 * T3 - 1.09331511e-11 * T4) *
              0.0399520575309628;
    // species 6: C2H2
    result += y[6] *
              (-1.91318906e-01 + 2.33615629e-02 * T - 3.55171815e-05 * T2 +
               2.80152437e-08 * T3 - 8.50072974e-12 * T4) *
              0.0384054074813734;
    // species 7: C2H3
    result += y[7] *
              (+2.21246645e+00 + 1.51479162e-03 * T + 2.59209412e-05 * T2 -
               3.57657847e-08 * T3 + 1.47150873e-11 * T4) *
              0.0369740442209569;
    // species 8: C2H4
    result += y[8] *
              (+2.95920148e+00 - 7.57052247e-03 * T + 5.70990292e-05 * T2 -
               6.91588753e-08 * T3 + 2.69884373e-11 * T4) *
              0.0356455407428531;
    // species 9: C2H5
    result += y[9] *
              (+3.30646568e+00 - 4.18658892e-03 * T + 4.97142807e-05 * T2 -
               5.99126606e-08 * T3 + 2.30509004e-11 * T4) *
              0.0344091941366733;
    // species 10: C2H6
    result += y[10] *
              (+3.29142492e+00 - 5.50154270e-03 * T + 5.99438288e-05 * T2 -
               7.08466285e-08 * T3 + 2.68685771e-11 * T4) *
              0.0332557366145660;
    // species 11: C(s)
    result += y[11] *
              (-1.31087207e+00 + 4.40353686e-03 * T + 1.90394118e-06 * T2 -
               6.38546966e-09 * T3 + 2.98964248e-12 * T4) *
              0.0832570144034635;
  } else {
    // species 0: H2
    result += y[0] *
              (+2.33727920e+00 - 4.94024731e-05 * T + 4.99456778e-07 * T2 -
               1.79566394e-10 * T3 + 2.00255376e-14 * T4) *
              0.4960317460317460;
    // species 1: H
    result += y[1] *
              (+1.50000001e+00 - 2.30842973e-11 * T + 1.61561948e-14 * T2 -
               4.73515235e-18 * T3 + 4.98197357e-22 * T4) *
              0.9920634920634921;
    // species 2: CH2
    result += y[2] *
              (+1.87410113e+00 + 3.65639292e-03 * T - 1.40894597e-06 * T2 +
               2.60179549e-10 * T3 - 1.87727567e-14 * T4) *
              0.0712910814857061;
    // species 3: CH3
    result += y[3] *
              (+1.28571772e+00 + 7.23990037e-03 * T - 2.98714348e-06 * T2 +
               5.95684644e-10 * T3 - 4.67154394e-14 * T4) *
              0.0665114732291320;
    // species 4: CH4
    result += y[4] *
              (-9.25148505e-01 + 1.33909467e-02 * T - 5.73285809e-06 * T2 +
               1.22292535e-09 * T3 - 1.01815230e-13 * T4) *
              0.0623324814560868;
    // species 5: C2H
    result += y[5] *
              (+2.16780652e+00 + 4.75221902e-03 * T - 1.83787077e-06 * T2 +
               3.04190252e-10 * T3 - 1.77232770e-14 * T4) *
              0.0399520575309628;
    // species 6: C2H2
    result += y[6] *
              (+3.14756964e+00 + 5.96166664e-03 * T - 2.37294852e-06 * T2 +
               4.67412171e-10 * T3 - 3.61235213e-14 * T4) *
              0.0384054074813734;
    // species 7: C2H3
    result += y[7] *
              (+2.01672400e+00 + 1.03302292e-02 * T - 4.68082349e-06 * T2 +
               1.01763288e-09 * T3 - 8.62607041e-14 * T4) *
              0.0369740442209569;
    // species 8: C2H4
    result += y[8] *
              (+1.03611116e+00 + 1.46454151e-02 * T - 6.71077915e-06 * T2 +
               1.47222923e-09 * T3 - 1.25706061e-13 * T4) *
              0.0356455407428531;
    // species 9: C2H5
    result += y[9] *
              (+9.54656420e-01 + 1.73972722e-02 * T - 7.98206668e-06 * T2 +
               1.75217689e-09 * T3 - 1.49641576e-13 * T4) *
              0.0344091941366733;
    // species 10: C2H6
    result += y[10] *
              (+7.18815000e-02 + 2.16852677e-02 * T - 1.00256067e-05 * T2 +
               2.21412001e-09 * T3 - 1.90002890e-13 * T4) *
              0.0332557366145660;
    // species 11: C(s)
    result += y[11] *
              (+4.55718290e-01 + 1.71702216e-03 * T - 6.97562786e-07 * T2 +
               1.35277032e-10 * T3 - 9.67590652e-15 * T4) *
              0.0832570144034635;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    result +=
      y[0] *
      (+2.34433112e+00 + 3.99026037e-03 * T - 6.49271700e-06 * T2 +
       5.03930235e-09 * T3 - 1.47522352e-12 * T4 - 9.17935173e+02 * invT) *
      0.4960317460317460;
    // species 1: H
    result +=
      y[1] *
      (+2.50000000e+00 + 3.52666409e-13 * T - 6.65306547e-16 * T2 +
       5.75204080e-19 * T3 - 1.85546466e-22 * T4 + 2.54736599e+04 * invT) *
      0.9920634920634921;
    // species 2: CH2
    result +=
      y[2] *
      (+3.76267867e+00 + 4.84436072e-04 * T + 9.31632803e-07 * T2 -
       9.62727883e-10 * T3 + 3.37483438e-13 * T4 + 4.60040401e+04 * invT) *
      0.0712910814857061;
    // species 3: CH3
    result +=
      y[3] *
      (+3.67359040e+00 + 1.00547588e-03 * T + 1.91007285e-06 * T2 -
       1.71779356e-09 * T3 + 5.08771468e-13 * T4 + 1.64449988e+04 * invT) *
      0.0665114732291320;
    // species 4: CH4
    result +=
      y[4] *
      (+5.14987613e+00 - 6.83548940e-03 * T + 1.63933533e-05 * T2 -
       1.21185757e-08 * T3 + 3.33387912e-12 * T4 - 1.02466476e+04 * invT) *
      0.0623324814560868;
    // species 5: C2H
    result +=
      y[5] *
      (+2.88965733e+00 + 6.70498055e-03 * T - 9.49231670e-06 * T2 +
       7.36977613e-09 * T3 - 2.18663022e-12 * T4 + 6.68393932e+04 * invT) *
      0.0399520575309628;
    // species 6: C2H2
    result +=
      y[6] *
      (+8.08681094e-01 + 1.16807815e-02 * T - 1.18390605e-05 * T2 +
       7.00381092e-09 * T3 - 1.70014595e-12 * T4 + 2.64289807e+04 * invT) *
      0.0384054074813734;
    // species 7: C2H3
    result +=
      y[7] *
      (+3.21246645e+00 + 7.57395810e-04 * T + 8.64031373e-06 * T2 -
       8.94144617e-09 * T3 + 2.94301746e-12 * T4 + 3.48598468e+04 * invT) *
      0.0369740442209569;
    // species 8: C2H4
    result +=
      y[8] *
      (+3.95920148e+00 - 3.78526124e-03 * T + 1.90330097e-05 * T2 -
       1.72897188e-08 * T3 + 5.39768746e-12 * T4 + 5.08977593e+03 * invT) *
      0.0356455407428531;
    // species 9: C2H5
    result +=
      y[9] *
      (+4.30646568e+00 - 2.09329446e-03 * T + 1.65714269e-05 * T2 -
       1.49781651e-08 * T3 + 4.61018008e-12 * T4 + 1.28416265e+04 * invT) *
      0.0344091941366733;
    // species 10: C2H6
    result +=
      y[10] *
      (+4.29142492e+00 - 2.75077135e-03 * T + 1.99812763e-05 * T2 -
       1.77116571e-08 * T3 + 5.37371542e-12 * T4 - 1.15222055e+04 * invT) *
      0.0332557366145660;
    // species 11: C(s)
    result +=
      y[11] *
      (-3.10872072e-01 + 2.20176843e-03 * T + 6.34647060e-07 * T2 -
       1.59636741e-09 * T3 + 5.97928496e-13 * T4 - 1.08650794e+02 * invT) *
      0.0832570144034635;
  } else {
    // species 0: H2
    result +=
      y[0] *
      (+3.33727920e+00 - 2.47012365e-05 * T + 1.66485593e-07 * T2 -
       4.48915985e-11 * T3 + 4.00510752e-15 * T4 - 9.50158922e+02 * invT) *
      0.4960317460317460;
    // species 1: H
    result +=
      y[1] *
      (+2.50000001e+00 - 1.15421486e-11 * T + 5.38539827e-15 * T2 -
       1.18378809e-18 * T3 + 9.96394714e-23 * T4 + 2.54736599e+04 * invT) *
      0.9920634920634921;
    // species 2: CH2
    result +=
      y[2] *
      (+2.87410113e+00 + 1.82819646e-03 * T - 4.69648657e-07 * T2 +
       6.50448872e-11 * T3 - 3.75455134e-15 * T4 + 4.62636040e+04 * invT) *
      0.0712910814857061;
    // species 3: CH3
    result +=
      y[3] *
      (+2.28571772e+00 + 3.61995018e-03 * T - 9.95714493e-07 * T2 +
       1.48921161e-10 * T3 - 9.34308788e-15 * T4 + 1.67755843e+04 * invT) *
      0.0665114732291320;
    // species 4: CH4
    result +=
      y[4] *
      (+7.48514950e-02 + 6.69547335e-03 * T - 1.91095270e-06 * T2 +
       3.05731338e-10 * T3 - 2.03630460e-14 * T4 - 9.46834459e+03 * invT) *
      0.0623324814560868;
    // species 5: C2H
    result +=
      y[5] *
      (+3.16780652e+00 + 2.37610951e-03 * T - 6.12623590e-07 * T2 +
       7.60475630e-11 * T3 - 3.54465540e-15 * T4 + 6.71210650e+04 * invT) *
      0.0399520575309628;
    // species 6: C2H2
    result +=
      y[6] *
      (+4.14756964e+00 + 2.98083332e-03 * T - 7.90982840e-07 * T2 +
       1.16853043e-10 * T3 - 7.22470426e-15 * T4 + 2.59359992e+04 * invT) *
      0.0384054074813734;
    // species 7: C2H3
    result +=
      y[7] *
      (+3.01672400e+00 + 5.16511460e-03 * T - 1.56027450e-06 * T2 +
       2.54408220e-10 * T3 - 1.72521408e-14 * T4 + 3.46128739e+04 * invT) *
      0.0369740442209569;
    // species 8: C2H4
    result +=
      y[8] *
      (+2.03611116e+00 + 7.32270755e-03 * T - 2.23692638e-06 * T2 +
       3.68057308e-10 * T3 - 2.51412122e-14 * T4 + 4.93988614e+03 * invT) *
      0.0356455407428531;
    // species 9: C2H5
    result +=
      y[9] *
      (+1.95465642e+00 + 8.69863610e-03 * T - 2.66068889e-06 * T2 +
       4.38044223e-10 * T3 - 2.99283152e-14 * T4 + 1.28575200e+04 * invT) *
      0.0344091941366733;
    // species 10: C2H6
    result +=
      y[10] *
      (+1.07188150e+00 + 1.08426339e-02 * T - 3.34186890e-06 * T2 +
       5.53530003e-10 * T3 - 3.80005780e-14 * T4 - 1.14263932e+04 * invT) *
      0.0332557366145660;
    // species 11: C(s)
    result +=
      y[11] *
      (+1.45571829e+00 + 8.58511080e-04 * T - 2.32520929e-07 * T2 +
       3.38192580e-11 * T3 - 1.93518130e-15 * T4 - 6.95138814e+02 * invT) *
      0.0832570144034635;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    result +=
      y[0] *
      (+1.34433112e+00 + 3.99026037e-03 * T - 6.49271700e-06 * T2 +
       5.03930235e-09 * T3 - 1.47522352e-12 * T4 - 9.17935173e+02 * invT) *
      0.4960317460317460;
    // species 1: H
    result +=
      y[1] *
      (+1.50000000e+00 + 3.52666409e-13 * T - 6.65306547e-16 * T2 +
       5.75204080e-19 * T3 - 1.85546466e-22 * T4 + 2.54736599e+04 * invT) *
      0.9920634920634921;
    // species 2: CH2
    result +=
      y[2] *
      (+2.76267867e+00 + 4.84436072e-04 * T + 9.31632803e-07 * T2 -
       9.62727883e-10 * T3 + 3.37483438e-13 * T4 + 4.60040401e+04 * invT) *
      0.0712910814857061;
    // species 3: CH3
    result +=
      y[3] *
      (+2.67359040e+00 + 1.00547588e-03 * T + 1.91007285e-06 * T2 -
       1.71779356e-09 * T3 + 5.08771468e-13 * T4 + 1.64449988e+04 * invT) *
      0.0665114732291320;
    // species 4: CH4
    result +=
      y[4] *
      (+4.14987613e+00 - 6.83548940e-03 * T + 1.63933533e-05 * T2 -
       1.21185757e-08 * T3 + 3.33387912e-12 * T4 - 1.02466476e+04 * invT) *
      0.0623324814560868;
    // species 5: C2H
    result +=
      y[5] *
      (+1.88965733e+00 + 6.70498055e-03 * T - 9.49231670e-06 * T2 +
       7.36977613e-09 * T3 - 2.18663022e-12 * T4 + 6.68393932e+04 * invT) *
      0.0399520575309628;
    // species 6: C2H2
    result +=
      y[6] *
      (-1.91318906e-01 + 1.16807815e-02 * T - 1.18390605e-05 * T2 +
       7.00381092e-09 * T3 - 1.70014595e-12 * T4 + 2.64289807e+04 * invT) *
      0.0384054074813734;
    // species 7: C2H3
    result +=
      y[7] *
      (+2.21246645e+00 + 7.57395810e-04 * T + 8.64031373e-06 * T2 -
       8.94144617e-09 * T3 + 2.94301746e-12 * T4 + 3.48598468e+04 * invT) *
      0.0369740442209569;
    // species 8: C2H4
    result +=
      y[8] *
      (+2.95920148e+00 - 3.78526124e-03 * T + 1.90330097e-05 * T2 -
       1.72897188e-08 * T3 + 5.39768746e-12 * T4 + 5.08977593e+03 * invT) *
      0.0356455407428531;
    // species 9: C2H5
    result +=
      y[9] *
      (+3.30646568e+00 - 2.09329446e-03 * T + 1.65714269e-05 * T2 -
       1.49781651e-08 * T3 + 4.61018008e-12 * T4 + 1.28416265e+04 * invT) *
      0.0344091941366733;
    // species 10: C2H6
    result +=
      y[10] *
      (+3.29142492e+00 - 2.75077135e-03 * T + 1.99812763e-05 * T2 -
       1.77116571e-08 * T3 + 5.37371542e-12 * T4 - 1.15222055e+04 * invT) *
      0.0332557366145660;
    // species 11: C(s)
    result +=
      y[11] *
      (-1.31087207e+00 + 2.20176843e-03 * T + 6.34647060e-07 * T2 -
       1.59636741e-09 * T3 + 5.97928496e-13 * T4 - 1.08650794e+02 * invT) *
      0.0832570144034635;
  } else {
    // species 0: H2
    result +=
      y[0] *
      (+2.33727920e+00 - 2.47012365e-05 * T + 1.66485593e-07 * T2 -
       4.48915985e-11 * T3 + 4.00510752e-15 * T4 - 9.50158922e+02 * invT) *
      0.4960317460317460;
    // species 1: H
    result +=
      y[1] *
      (+1.50000001e+00 - 1.15421486e-11 * T + 5.38539827e-15 * T2 -
       1.18378809e-18 * T3 + 9.96394714e-23 * T4 + 2.54736599e+04 * invT) *
      0.9920634920634921;
    // species 2: CH2
    result +=
      y[2] *
      (+1.87410113e+00 + 1.82819646e-03 * T - 4.69648657e-07 * T2 +
       6.50448872e-11 * T3 - 3.75455134e-15 * T4 + 4.62636040e+04 * invT) *
      0.0712910814857061;
    // species 3: CH3
    result +=
      y[3] *
      (+1.28571772e+00 + 3.61995018e-03 * T - 9.95714493e-07 * T2 +
       1.48921161e-10 * T3 - 9.34308788e-15 * T4 + 1.67755843e+04 * invT) *
      0.0665114732291320;
    // species 4: CH4
    result +=
      y[4] *
      (-9.25148505e-01 + 6.69547335e-03 * T - 1.91095270e-06 * T2 +
       3.05731338e-10 * T3 - 2.03630460e-14 * T4 - 9.46834459e+03 * invT) *
      0.0623324814560868;
    // species 5: C2H
    result +=
      y[5] *
      (+2.16780652e+00 + 2.37610951e-03 * T - 6.12623590e-07 * T2 +
       7.60475630e-11 * T3 - 3.54465540e-15 * T4 + 6.71210650e+04 * invT) *
      0.0399520575309628;
    // species 6: C2H2
    result +=
      y[6] *
      (+3.14756964e+00 + 2.98083332e-03 * T - 7.90982840e-07 * T2 +
       1.16853043e-10 * T3 - 7.22470426e-15 * T4 + 2.59359992e+04 * invT) *
      0.0384054074813734;
    // species 7: C2H3
    result +=
      y[7] *
      (+2.01672400e+00 + 5.16511460e-03 * T - 1.56027450e-06 * T2 +
       2.54408220e-10 * T3 - 1.72521408e-14 * T4 + 3.46128739e+04 * invT) *
      0.0369740442209569;
    // species 8: C2H4
    result +=
      y[8] *
      (+1.03611116e+00 + 7.32270755e-03 * T - 2.23692638e-06 * T2 +
       3.68057308e-10 * T3 - 2.51412122e-14 * T4 + 4.93988614e+03 * invT) *
      0.0356455407428531;
    // species 9: C2H5
    result +=
      y[9] *
      (+9.54656420e-01 + 8.69863610e-03 * T - 2.66068889e-06 * T2 +
       4.38044223e-10 * T3 - 2.99283152e-14 * T4 + 1.28575200e+04 * invT) *
      0.0344091941366733;
    // species 10: C2H6
    result +=
      y[10] *
      (+7.18815000e-02 + 1.08426339e-02 * T - 3.34186890e-06 * T2 +
       5.53530003e-10 * T3 - 3.80005780e-14 * T4 - 1.14263932e+04 * invT) *
      0.0332557366145660;
    // species 11: C(s)
    result +=
      y[11] *
      (+4.55718290e-01 + 8.58511080e-04 * T - 2.32520929e-07 * T2 +
       3.38192580e-11 * T3 - 1.93518130e-15 * T4 - 6.95138814e+02 * invT) *
      0.0832570144034635;
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
  x[0] = y[0] / (2.0160000000000000e+00 * YOW);
  x[1] = y[1] / (1.0080000000000000e+00 * YOW);
  x[2] = y[2] / (1.4026999999999999e+01 * YOW);
  x[3] = y[3] / (1.5035000000000000e+01 * YOW);
  x[4] = y[4] / (1.6042999999999999e+01 * YOW);
  x[5] = y[5] / (2.5029999999999998e+01 * YOW);
  x[6] = y[6] / (2.6037999999999997e+01 * YOW);
  x[7] = y[7] / (2.7045999999999999e+01 * YOW);
  x[8] = y[8] / (2.8053999999999998e+01 * YOW);
  x[9] = y[9] / (2.9061999999999998e+01 * YOW);
  x[10] = y[10] / (3.0070000000000000e+01 * YOW);
  x[11] = y[11] / (1.2010999999999999e+01 * YOW);
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
  amrex::Real tmax = 3500.0; // min upper bound for thermo def
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
  amrex::Real tmax = 3500.0; // min upper bound for thermo def
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
  XW += x[0] * 2.0160000000000000e+00;     // H2
  XW += x[1] * 1.0080000000000000e+00;     // H
  XW += x[2] * 1.4026999999999999e+01;     // CH2
  XW += x[3] * 1.5035000000000000e+01;     // CH3
  XW += x[4] * 1.6042999999999999e+01;     // CH4
  XW += x[5] * 2.5029999999999998e+01;     // C2H
  XW += x[6] * 2.6037999999999997e+01;     // C2H2
  XW += x[7] * 2.7045999999999999e+01;     // C2H3
  XW += x[8] * 2.8053999999999998e+01;     // C2H4
  XW += x[9] * 2.9061999999999998e+01;     // C2H5
  XW += x[10] * 3.0070000000000000e+01;    // C2H6
  XW += x[11] * 1.2010999999999999e+01;    // C(s)
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
  W += c[0] * 2.0160000000000000e+00;  // H2
  W += c[1] * 1.0080000000000000e+00;  // H
  W += c[2] * 1.4026999999999999e+01;  // CH2
  W += c[3] * 1.5035000000000000e+01;  // CH3
  W += c[4] * 1.6042999999999999e+01;  // CH4
  W += c[5] * 2.5029999999999998e+01;  // C2H
  W += c[6] * 2.6037999999999997e+01;  // C2H2
  W += c[7] * 2.7045999999999999e+01;  // C2H3
  W += c[8] * 2.8053999999999998e+01;  // C2H4
  W += c[9] * 2.9061999999999998e+01;  // C2H5
  W += c[10] * 3.0070000000000000e+01; // C2H6
  W += c[11] * 1.2010999999999999e+01; // C(s)

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
  XW += x[0] * 2.0160000000000000e+00;       // H2
  XW += x[1] * 1.0080000000000000e+00;       // H
  XW += x[2] * 1.4026999999999999e+01;       // CH2
  XW += x[3] * 1.5035000000000000e+01;       // CH3
  XW += x[4] * 1.6042999999999999e+01;       // CH4
  XW += x[5] * 2.5029999999999998e+01;       // C2H
  XW += x[6] * 2.6037999999999997e+01;       // C2H2
  XW += x[7] * 2.7045999999999999e+01;       // C2H3
  XW += x[8] * 2.8053999999999998e+01;       // C2H4
  XW += x[9] * 2.9061999999999998e+01;       // C2H5
  XW += x[10] * 3.0070000000000000e+01;      // C2H6
  XW += x[11] * 1.2010999999999999e+01;      // C(s)
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
  W += c[0] * 2.0160000000000000e+00;  // H2
  W += c[1] * 1.0080000000000000e+00;  // H
  W += c[2] * 1.4026999999999999e+01;  // CH2
  W += c[3] * 1.5035000000000000e+01;  // CH3
  W += c[4] * 1.6042999999999999e+01;  // CH4
  W += c[5] * 2.5029999999999998e+01;  // C2H
  W += c[6] * 2.6037999999999997e+01;  // C2H2
  W += c[7] * 2.7045999999999999e+01;  // C2H3
  W += c[8] * 2.8053999999999998e+01;  // C2H4
  W += c[9] * 2.9061999999999998e+01;  // C2H5
  W += c[10] * 3.0070000000000000e+01; // C2H6
  W += c[11] * 1.2010999999999999e+01; // C(s)

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
  XW += x[0] * 2.0160000000000000e+00;  // H2
  XW += x[1] * 1.0080000000000000e+00;  // H
  XW += x[2] * 1.4026999999999999e+01;  // CH2
  XW += x[3] * 1.5035000000000000e+01;  // CH3
  XW += x[4] * 1.6042999999999999e+01;  // CH4
  XW += x[5] * 2.5029999999999998e+01;  // C2H
  XW += x[6] * 2.6037999999999997e+01;  // C2H2
  XW += x[7] * 2.7045999999999999e+01;  // C2H3
  XW += x[8] * 2.8053999999999998e+01;  // C2H4
  XW += x[9] * 2.9061999999999998e+01;  // C2H5
  XW += x[10] * 3.0070000000000000e+01; // C2H6
  XW += x[11] * 1.2010999999999999e+01; // C(s)
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
  W += c[0] * 2.0160000000000000e+00;  // H2
  W += c[1] * 1.0080000000000000e+00;  // H
  W += c[2] * 1.4026999999999999e+01;  // CH2
  W += c[3] * 1.5035000000000000e+01;  // CH3
  W += c[4] * 1.6042999999999999e+01;  // CH4
  W += c[5] * 2.5029999999999998e+01;  // C2H
  W += c[6] * 2.6037999999999997e+01;  // C2H2
  W += c[7] * 2.7045999999999999e+01;  // C2H3
  W += c[8] * 2.8053999999999998e+01;  // C2H4
  W += c[9] * 2.9061999999999998e+01;  // C2H5
  W += c[10] * 3.0070000000000000e+01; // C2H6
  W += c[11] * 1.2010999999999999e+01; // C(s)

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
  XW += x[0] * 2.0160000000000000e+00;  // H2
  XW += x[1] * 1.0080000000000000e+00;  // H
  XW += x[2] * 1.4026999999999999e+01;  // CH2
  XW += x[3] * 1.5035000000000000e+01;  // CH3
  XW += x[4] * 1.6042999999999999e+01;  // CH4
  XW += x[5] * 2.5029999999999998e+01;  // C2H
  XW += x[6] * 2.6037999999999997e+01;  // C2H2
  XW += x[7] * 2.7045999999999999e+01;  // C2H3
  XW += x[8] * 2.8053999999999998e+01;  // C2H4
  XW += x[9] * 2.9061999999999998e+01;  // C2H5
  XW += x[10] * 3.0070000000000000e+01; // C2H6
  XW += x[11] * 1.2010999999999999e+01; // C(s)
  // Now compute conversion
  amrex::Real XWinv = 1.0 / XW;
  y[0] = x[0] * 2.0160000000000000e+00 * XWinv;
  y[1] = x[1] * 1.0080000000000000e+00 * XWinv;
  y[2] = x[2] * 1.4026999999999999e+01 * XWinv;
  y[3] = x[3] * 1.5035000000000000e+01 * XWinv;
  y[4] = x[4] * 1.6042999999999999e+01 * XWinv;
  y[5] = x[5] * 2.5029999999999998e+01 * XWinv;
  y[6] = x[6] * 2.6037999999999997e+01 * XWinv;
  y[7] = x[7] * 2.7045999999999999e+01 * XWinv;
  y[8] = x[8] * 2.8053999999999998e+01 * XWinv;
  y[9] = x[9] * 2.9061999999999998e+01 * XWinv;
  y[10] = x[10] * 3.0070000000000000e+01 * XWinv;
  y[11] = x[11] * 1.2010999999999999e+01 * XWinv;
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
  XW += x[0] * 2.0160000000000000e+00;  // H2
  XW += x[1] * 1.0080000000000000e+00;  // H
  XW += x[2] * 1.4026999999999999e+01;  // CH2
  XW += x[3] * 1.5035000000000000e+01;  // CH3
  XW += x[4] * 1.6042999999999999e+01;  // CH4
  XW += x[5] * 2.5029999999999998e+01;  // C2H
  XW += x[6] * 2.6037999999999997e+01;  // C2H2
  XW += x[7] * 2.7045999999999999e+01;  // C2H3
  XW += x[8] * 2.8053999999999998e+01;  // C2H4
  XW += x[9] * 2.9061999999999998e+01;  // C2H5
  XW += x[10] * 3.0070000000000000e+01; // C2H6
  XW += x[11] * 1.2010999999999999e+01; // C(s)
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
  CW += c[0] * 2.0160000000000000e+00;  // H2
  CW += c[1] * 1.0080000000000000e+00;  // H
  CW += c[2] * 1.4026999999999999e+01;  // CH2
  CW += c[3] * 1.5035000000000000e+01;  // CH3
  CW += c[4] * 1.6042999999999999e+01;  // CH4
  CW += c[5] * 2.5029999999999998e+01;  // C2H
  CW += c[6] * 2.6037999999999997e+01;  // C2H2
  CW += c[7] * 2.7045999999999999e+01;  // C2H3
  CW += c[8] * 2.8053999999999998e+01;  // C2H4
  CW += c[9] * 2.9061999999999998e+01;  // C2H5
  CW += c[10] * 3.0070000000000000e+01; // C2H6
  CW += c[11] * 1.2010999999999999e+01; // C(s)
  // Now compute conversion
  amrex::Real CWinv = 1.0 / CW;
  y[0] = c[0] * 2.0160000000000000e+00 * CWinv;
  y[1] = c[1] * 1.0080000000000000e+00 * CWinv;
  y[2] = c[2] * 1.4026999999999999e+01 * CWinv;
  y[3] = c[3] * 1.5035000000000000e+01 * CWinv;
  y[4] = c[4] * 1.6042999999999999e+01 * CWinv;
  y[5] = c[5] * 2.5029999999999998e+01 * CWinv;
  y[6] = c[6] * 2.6037999999999997e+01 * CWinv;
  y[7] = c[7] * 2.7045999999999999e+01 * CWinv;
  y[8] = c[8] * 2.8053999999999998e+01 * CWinv;
  y[9] = c[9] * 2.9061999999999998e+01 * CWinv;
  y[10] = c[10] * 3.0070000000000000e+01 * CWinv;
  y[11] = c[11] * 1.2010999999999999e+01 * CWinv;
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
  cvms[0] *= 4.124237409798234e+07;  // H2
  cvms[1] *= 8.248474819596468e+07;  // H
  cvms[2] *= 5.927470320206203e+06;  // CH2
  cvms[3] *= 5.530071578419182e+06;  // CH3
  cvms[4] *= 5.182610869633635e+06;  // CH4
  cvms[5] *= 3.321798888594982e+06;  // C2H
  cvms[6] *= 3.193203248388218e+06;  // C2H2
  cvms[7] *= 3.074193085170909e+06;  // C2H3
  cvms[8] *= 2.963735160103101e+06;  // C2H4
  cvms[9] *= 2.860939583701480e+06;  // C2H5
  cvms[10] *= 2.765035789209591e+06; // C2H6
  cvms[11] *= 6.922373339566431e+06; // C(s)
}

// Returns the specific heats at constant pressure
// in mass units (Eq. 26)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPMS(const amrex::Real T, amrex::Real cpms[])
{
  cp_R(cpms, T);
  // multiply by R/molecularweight
  cpms[0] *= 4.124237409798234e+07;  // H2
  cpms[1] *= 8.248474819596468e+07;  // H
  cpms[2] *= 5.927470320206203e+06;  // CH2
  cpms[3] *= 5.530071578419182e+06;  // CH3
  cpms[4] *= 5.182610869633635e+06;  // CH4
  cpms[5] *= 3.321798888594982e+06;  // C2H
  cpms[6] *= 3.193203248388218e+06;  // C2H2
  cpms[7] *= 3.074193085170909e+06;  // C2H3
  cpms[8] *= 2.963735160103101e+06;  // C2H4
  cpms[9] *= 2.860939583701480e+06;  // C2H5
  cpms[10] *= 2.765035789209591e+06; // C2H6
  cpms[11] *= 6.922373339566431e+06; // C(s)
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
  sms[0] *= 4.124237409798234e+07;  // H2
  sms[1] *= 8.248474819596468e+07;  // H
  sms[2] *= 5.927470320206203e+06;  // CH2
  sms[3] *= 5.530071578419182e+06;  // CH3
  sms[4] *= 5.182610869633635e+06;  // CH4
  sms[5] *= 3.321798888594982e+06;  // C2H
  sms[6] *= 3.193203248388218e+06;  // C2H2
  sms[7] *= 3.074193085170909e+06;  // C2H3
  sms[8] *= 2.963735160103101e+06;  // C2H4
  sms[9] *= 2.860939583701480e+06;  // C2H5
  sms[10] *= 2.765035789209591e+06; // C2H6
  sms[11] *= 6.922373339566431e+06; // C(s)
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

  // reaction 0: CH3 + H + M <=> CH4 + M
  qf[0] = sc[1] * sc[3];
  qr[0] = sc[4];

  // reaction 3: C2H4 + H (+M) <=> C2H5 (+M)
  qf[1] = sc[1] * sc[8];
  qr[1] = sc[9];

  // reaction 5: C2H2 + H (+M) <=> C2H3 (+M)
  qf[2] = sc[1] * sc[6];
  qr[2] = sc[7];

  // reaction 6: C2H + H (+M) <=> C2H2 (+M)
  qf[3] = sc[1] * sc[5];
  qr[3] = sc[6];

  // reaction 10: C2H4 + M <=> C2H2 + H2 + M
  qf[4] = sc[8];
  qr[4] = sc[0] * sc[6];

  // reaction 12: 2 CH3 + M <=> C2H6 + M
  qf[5] = (sc[3] * sc[3]);
  qr[5] = sc[10];

  // reaction 1: CH4 + H <=> CH3 + H2
  qf[6] = sc[1] * sc[4];
  qr[6] = sc[0] * sc[3];

  // reaction 2: 2 CH3 <=> C2H5 + H
  qf[7] = (sc[3] * sc[3]);
  qr[7] = sc[1] * sc[9];

  // reaction 4: C2H4 + H <=> C2H3 + H2
  qf[8] = sc[1] * sc[8];
  qr[8] = sc[0] * sc[7];

  // reaction 7: C2H + H2 <=> C2H2 + H
  qf[9] = sc[0] * sc[5];
  qr[9] = sc[1] * sc[6];

  // reaction 8: CH2 + H2 <=> CH3 + H
  qf[10] = sc[0] * sc[2];
  qr[10] = sc[1] * sc[3];

  // reaction 9: CH2 + CH3 <=> C2H4 + H
  qf[11] = sc[2] * sc[3];
  qr[11] = sc[1] * sc[8];

  // reaction 11: C2H2 <=> 2 C(s) + H2
  qf[12] = sc[6];
  qr[12] = sc[0] * (sc[11] * sc[11]);

  // reaction 13: C2H6 + H <=> C2H5 + H2
  qf[13] = sc[1] * sc[10];
  qr[13] = sc[0] * sc[9];

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

  // reaction 0:  CH3 + H + M <=> CH4 + M
  k_f = 18529779629.8165 *
        exp((-1.19245049206076) * logT - (85.2283723825864) * invT);
  Corr = mixture;
  qf[0] *= Corr * k_f;
  qr[0] *= Corr * k_f * exp(-(g_RT[1] + g_RT[3] - g_RT[4])) * (refC);
  // reaction 1:  CH4 + H <=> CH3 + H2
  k_f = 0.352683948427848 *
        exp((2.76711916768222) * logT - (8314.80590579313) * invT);
  qf[6] *= k_f;
  qr[6] *= k_f * exp(-(-g_RT[0] + g_RT[1] - g_RT[3] + g_RT[4]));
  // reaction 2:  2 CH3 <=> C2H5 + H
  k_f = 36152.1969455548 *
        exp((0.845120206098996) * logT - (3909.67459913287) * invT);
  qf[7] *= k_f;
  qr[7] *= k_f * exp(-(-g_RT[1] + 2.000000 * g_RT[3] - g_RT[9]));
  // reaction 3:  C2H4 + H (+M) <=> C2H5 (+M)
  k_f = 9867.16302841882 *
        exp((1.00304123676575) * logT - (1297.74828748432) * invT);
  Corr = mixture;
  qf[1] *= Corr * k_f;
  qr[1] *= Corr * k_f * exp(-(g_RT[1] + g_RT[8] - g_RT[9])) * (refC);
  // reaction 4:  C2H4 + H <=> C2H3 + H2
  k_f = 2512.05632509109 *
        exp((1.53801468328155) * logT - (8383.73163879634) * invT);
  qf[8] *= k_f;
  qr[8] *= k_f * exp(-(-g_RT[0] + g_RT[1] - g_RT[7] + g_RT[8]));
  // reaction 5:  C2H2 + H (+M) <=> C2H3 (+M)
  k_f = 68388.8864690645 *
        exp((-0.86014726711066) * logT - (1267.69380866152) * invT);
  Corr = mixture;
  qf[2] *= Corr * k_f;
  qr[2] *= Corr * k_f * exp(-(g_RT[1] + g_RT[6] - g_RT[7])) * (refC);
  // reaction 6:  C2H + H (+M) <=> C2H2 (+M)
  k_f = 4668349273.45473 *
        exp((-0.803167128908363) * logT - (0.00158333685917708) * invT);
  Corr = mixture;
  qf[3] *= Corr * k_f;
  qr[3] *= Corr * k_f * exp(-(g_RT[1] + g_RT[5] - g_RT[6])) * (refC);
  // reaction 7:  C2H + H2 <=> C2H2 + H
  k_f =
    5372180.16519361 * exp((1.6080006615735) * logT - (15642.633813977) * invT);
  qf[9] *= k_f;
  qr[9] *= k_f * exp(-(g_RT[0] - g_RT[1] + g_RT[5] - g_RT[6]));
  // reaction 8:  CH2 + H2 <=> CH3 + H
  k_f = 100.17524517356 *
        exp((2.68606997955989) * logT - (5251.06261291261) * invT);
  qf[10] *= k_f;
  qr[10] *= k_f * exp(-(g_RT[0] - g_RT[1] + g_RT[2] - g_RT[3]));
  // reaction 9:  CH2 + CH3 <=> C2H4 + H
  k_f = 4.0153594627338 *
        exp((0.575581019159047) * logT - (0.443945260758714) * invT);
  qf[11] *= k_f;
  qr[11] *= k_f * exp(-(-g_RT[1] + g_RT[2] + g_RT[3] - g_RT[8]));
  // reaction 10:  C2H4 + M <=> C2H2 + H2 + M
  k_f = 49191261.0996573 *
        exp((0.794734252072147) * logT - (31648.5563093388) * invT);
  Corr = mixture;
  qf[4] *= Corr * k_f;
  qr[4] *= Corr * k_f * exp(-(-g_RT[0] - g_RT[6] + g_RT[8])) * (refCinv);
  // reaction 11:  C2H2 <=> 2 C(s) + H2
  k_f = 123.472588382411 *
        exp((-0.293562938580551) * logT - (5.10231730398989) * invT);
  qf[12] *= k_f;
  qr[12] *= k_f * exp(-(-g_RT[0] + g_RT[6] - 2.000000 * g_RT[11])) *
            ((refCinv * refCinv));
  // reaction 12:  2 CH3 + M <=> C2H6 + M
  k_f = 1026575730779.29 *
        exp((-0.592670538495204) * logT - (642.311112548238) * invT);
  Corr = mixture;
  qf[5] *= Corr * k_f;
  qr[5] *= Corr * k_f * exp(-(2.000000 * g_RT[3] - g_RT[10])) * (refC);
  // reaction 13:  C2H6 + H <=> C2H5 + H2
  k_f = 2853873.1839057 *
        exp((1.43294743399092) * logT - (12431.3149961588) * invT);
  qf[13] *= k_f;
  qr[13] *= k_f * exp(-(-g_RT[0] + g_RT[1] - g_RT[9] + g_RT[10]));
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
    // reaction 0:  CH3 + H + M <=> CH4 + M
    const amrex::Real k_f =
      18529779629.8165 *
      exp((-1.19245049206076) * logT - (85.2283723825864) * invT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[3]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[3] - g_RT[4])) * (refC) * (sc[4]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;
  }

  {
    // reaction 3:  C2H4 + H (+M) <=> C2H5 (+M)
    const amrex::Real k_f =
      9867.16302841882 *
      exp((1.00304123676575) * logT - (1297.74828748432) * invT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[8]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[8] - g_RT[9])) * (refC) * (sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[8] -= qdot;
    wdot[9] += qdot;
  }

  {
    // reaction 5:  C2H2 + H (+M) <=> C2H3 (+M)
    const amrex::Real k_f =
      68388.8864690645 *
      exp((-0.86014726711066) * logT - (1267.69380866152) * invT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[6]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[6] - g_RT[7])) * (refC) * (sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[6] -= qdot;
    wdot[7] += qdot;
  }

  {
    // reaction 6:  C2H + H (+M) <=> C2H2 (+M)
    const amrex::Real k_f =
      4668349273.45473 *
      exp((-0.803167128908363) * logT - (0.00158333685917708) * invT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[5]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[5] - g_RT[6])) * (refC) * (sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[5] -= qdot;
    wdot[6] += qdot;
  }

  {
    // reaction 10:  C2H4 + M <=> C2H2 + H2 + M
    const amrex::Real k_f =
      49191261.0996573 *
      exp((0.794734252072147) * logT - (31648.5563093388) * invT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[8]);
    const amrex::Real qr = Corr * k_f * exp(-(-g_RT[0] - g_RT[6] + g_RT[8])) *
                           (refCinv) * (sc[0] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[6] += qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 12:  2 CH3 + M <=> C2H6 + M
    const amrex::Real k_f =
      1026575730779.29 *
      exp((-0.592670538495204) * logT - (642.311112548238) * invT);
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * ((sc[3] * sc[3]));
    const amrex::Real qr =
      Corr * k_f * exp(-(2.000000 * g_RT[3] - g_RT[10])) * (refC) * (sc[10]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= 2.000000 * qdot;
    wdot[10] += qdot;
  }

  {
    // reaction 1:  CH4 + H <=> CH3 + H2
    const amrex::Real k_f =
      0.352683948427848 *
      exp((2.76711916768222) * logT - (8314.80590579313) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] + g_RT[1] - g_RT[3] + g_RT[4])) * (sc[0] * sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[4] -= qdot;
  }

  {
    // reaction 2:  2 CH3 <=> C2H5 + H
    const amrex::Real k_f =
      36152.1969455548 *
      exp((0.845120206098996) * logT - (3909.67459913287) * invT);
    const amrex::Real qf = k_f * ((sc[3] * sc[3]));
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + 2.000000 * g_RT[3] - g_RT[9])) * (sc[1] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[3] -= 2.000000 * qdot;
    wdot[9] += qdot;
  }

  {
    // reaction 4:  C2H4 + H <=> C2H3 + H2
    const amrex::Real k_f =
      2512.05632509109 *
      exp((1.53801468328155) * logT - (8383.73163879634) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[8]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] + g_RT[1] - g_RT[7] + g_RT[8])) * (sc[0] * sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[1] -= qdot;
    wdot[7] += qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 7:  C2H + H2 <=> C2H2 + H
    const amrex::Real k_f =
      5372180.16519361 *
      exp((1.6080006615735) * logT - (15642.633813977) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[5]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] - g_RT[1] + g_RT[5] - g_RT[6])) * (sc[1] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[1] += qdot;
    wdot[5] -= qdot;
    wdot[6] += qdot;
  }

  {
    // reaction 8:  CH2 + H2 <=> CH3 + H
    const amrex::Real k_f =
      100.17524517356 *
      exp((2.68606997955989) * logT - (5251.06261291261) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[2]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] - g_RT[1] + g_RT[2] - g_RT[3])) * (sc[1] * sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[3] += qdot;
  }

  {
    // reaction 9:  CH2 + CH3 <=> C2H4 + H
    const amrex::Real k_f =
      4.0153594627338 *
      exp((0.575581019159047) * logT - (0.443945260758714) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[3]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[2] + g_RT[3] - g_RT[8])) * (sc[1] * sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[3] -= qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 11:  C2H2 <=> 2 C(s) + H2
    const amrex::Real k_f =
      123.472588382411 *
      exp((-0.293562938580551) * logT - (5.10231730398989) * invT);
    const amrex::Real qf = k_f * (sc[6]);
    const amrex::Real qr = k_f *
                           exp(-(-g_RT[0] + g_RT[6] - 2.000000 * g_RT[11])) *
                           ((refCinv * refCinv)) * (sc[0] * (sc[11] * sc[11]));
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[6] -= qdot;
    wdot[11] += 2.000000 * qdot;
  }

  {
    // reaction 13:  C2H6 + H <=> C2H5 + H2
    const amrex::Real k_f =
      2853873.1839057 *
      exp((1.43294743399092) * logT - (12431.3149961588) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[10]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] + g_RT[1] - g_RT[9] + g_RT[10])) * (sc[0] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[1] -= qdot;
    wdot[9] += qdot;
    wdot[10] -= qdot;
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
  XW += x[0] * 2.0160000000000000e+00;  // H2
  XW += x[1] * 1.0080000000000000e+00;  // H
  XW += x[2] * 1.4026999999999999e+01;  // CH2
  XW += x[3] * 1.5035000000000000e+01;  // CH3
  XW += x[4] * 1.6042999999999999e+01;  // CH4
  XW += x[5] * 2.5029999999999998e+01;  // C2H
  XW += x[6] * 2.6037999999999997e+01;  // C2H2
  XW += x[7] * 2.7045999999999999e+01;  // C2H3
  XW += x[8] * 2.8053999999999998e+01;  // C2H4
  XW += x[9] * 2.9061999999999998e+01;  // C2H5
  XW += x[10] * 3.0070000000000000e+01; // C2H6
  XW += x[11] * 1.2010999999999999e+01; // C(s)
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
  kcharge[0] = 0;  // H2
  kcharge[1] = 0;  // H
  kcharge[2] = 0;  // CH2
  kcharge[3] = 0;  // CH3
  kcharge[4] = 0;  // CH4
  kcharge[5] = 0;  // C2H
  kcharge[6] = 0;  // C2H2
  kcharge[7] = 0;  // C2H3
  kcharge[8] = 0;  // C2H4
  kcharge[9] = 0;  // C2H5
  kcharge[10] = 0; // C2H6
  kcharge[11] = 0; // C(s)
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
  // reaction 0: CH3 + H + M <=> CH4 + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[1] * sc[3];
  k_f = 18529779629.8165 *
        exp(-1.19245049206076 * logT - (85.2283723825864) * invT);
  dlnkfdT = -1.19245049206076 * invT + (85.2283723825864) * invT2;
  // reverse
  phi_r = sc[4];
  Kc = refCinv * exp(g_RT[1] + g_RT[3] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[3]) + (h_RT[4]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // H
  wdot[3] -= q; // CH3
  wdot[4] += q; // CH4
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor + k_f * sc[3];
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor + k_f * sc[1];
  dqdc[4] = q_nocor - k_r;
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  dqdc[9] = q_nocor;
  dqdc[10] = q_nocor;
  dqdc[11] = q_nocor;
  for (int k = 0; k < 12; k++) {
    J[13 * k + 1] -= dqdc[k];
    J[13 * k + 3] -= dqdc[k];
    J[13 * k + 4] += dqdc[k];
  }
  J[157] -= dqdT; // dwdot[H]/dT
  J[159] -= dqdT; // dwdot[CH3]/dT
  J[160] += dqdT; // dwdot[CH4]/dT

  // reaction 3: C2H4 + H (+M) <=> C2H5 (+M)
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[1] * sc[8];
  k_f =
    9867.16302841882 * exp(1.00304123676575 * logT - (1297.74828748432) * invT);
  dlnkfdT = 1.00304123676575 * invT + (1297.74828748432) * invT2;
  // reverse
  phi_r = sc[9];
  Kc = refCinv * exp(g_RT[1] + g_RT[8] - g_RT[9]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[8]) + (h_RT[9]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // H
  wdot[8] -= q; // C2H4
  wdot[9] += q; // C2H5
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor + k_f * sc[8];
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor;
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor + k_f * sc[1];
  dqdc[9] = q_nocor - k_r;
  dqdc[10] = q_nocor;
  dqdc[11] = q_nocor;
  for (int k = 0; k < 12; k++) {
    J[13 * k + 1] -= dqdc[k];
    J[13 * k + 8] -= dqdc[k];
    J[13 * k + 9] += dqdc[k];
  }
  J[157] -= dqdT; // dwdot[H]/dT
  J[164] -= dqdT; // dwdot[C2H4]/dT
  J[165] += dqdT; // dwdot[C2H5]/dT

  // reaction 5: C2H2 + H (+M) <=> C2H3 (+M)
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[1] * sc[6];
  k_f = 68388.8864690645 *
        exp(-0.86014726711066 * logT - (1267.69380866152) * invT);
  dlnkfdT = -0.86014726711066 * invT + (1267.69380866152) * invT2;
  // reverse
  phi_r = sc[7];
  Kc = refCinv * exp(g_RT[1] + g_RT[6] - g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[6]) + (h_RT[7]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // H
  wdot[6] -= q; // C2H2
  wdot[7] += q; // C2H3
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor + k_f * sc[6];
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor;
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor + k_f * sc[1];
  dqdc[7] = q_nocor - k_r;
  dqdc[8] = q_nocor;
  dqdc[9] = q_nocor;
  dqdc[10] = q_nocor;
  dqdc[11] = q_nocor;
  for (int k = 0; k < 12; k++) {
    J[13 * k + 1] -= dqdc[k];
    J[13 * k + 6] -= dqdc[k];
    J[13 * k + 7] += dqdc[k];
  }
  J[157] -= dqdT; // dwdot[H]/dT
  J[162] -= dqdT; // dwdot[C2H2]/dT
  J[163] += dqdT; // dwdot[C2H3]/dT

  // reaction 6: C2H + H (+M) <=> C2H2 (+M)
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[1] * sc[5];
  k_f = 4668349273.45473 *
        exp(-0.803167128908363 * logT - (0.00158333685917708) * invT);
  dlnkfdT = -0.803167128908363 * invT + (0.00158333685917708) * invT2;
  // reverse
  phi_r = sc[6];
  Kc = refCinv * exp(g_RT[1] + g_RT[5] - g_RT[6]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[5]) + (h_RT[6]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // H
  wdot[5] -= q; // C2H
  wdot[6] += q; // C2H2
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor + k_f * sc[5];
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor;
  dqdc[5] = q_nocor + k_f * sc[1];
  dqdc[6] = q_nocor - k_r;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  dqdc[9] = q_nocor;
  dqdc[10] = q_nocor;
  dqdc[11] = q_nocor;
  for (int k = 0; k < 12; k++) {
    J[13 * k + 1] -= dqdc[k];
    J[13 * k + 5] -= dqdc[k];
    J[13 * k + 6] += dqdc[k];
  }
  J[157] -= dqdT; // dwdot[H]/dT
  J[161] -= dqdT; // dwdot[C2H]/dT
  J[162] += dqdT; // dwdot[C2H2]/dT

  // reaction 10: C2H4 + M <=> C2H2 + H2 + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[8];
  k_f = 49191261.0996573 *
        exp(0.794734252072147 * logT - (31648.5563093388) * invT);
  dlnkfdT = 0.794734252072147 * invT + (31648.5563093388) * invT2;
  // reverse
  phi_r = sc[0] * sc[6];
  Kc = refC * exp(-g_RT[0] - g_RT[6] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8]) + (h_RT[0] + h_RT[6]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // H2
  wdot[6] += q; // C2H2
  wdot[8] -= q; // C2H4
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor - k_r * sc[6];
  dqdc[1] = q_nocor;
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor;
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor - k_r * sc[0];
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor + k_f;
  dqdc[9] = q_nocor;
  dqdc[10] = q_nocor;
  dqdc[11] = q_nocor;
  for (int k = 0; k < 12; k++) {
    J[13 * k + 0] += dqdc[k];
    J[13 * k + 6] += dqdc[k];
    J[13 * k + 8] -= dqdc[k];
  }
  J[156] += dqdT; // dwdot[H2]/dT
  J[162] += dqdT; // dwdot[C2H2]/dT
  J[164] -= dqdT; // dwdot[C2H4]/dT

  // reaction 12: 2 CH3 + M <=> C2H6 + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = (sc[3] * sc[3]);
  k_f = 1026575730779.29 *
        exp(-0.592670538495204 * logT - (642.311112548238) * invT);
  dlnkfdT = -0.592670538495204 * invT + (642.311112548238) * invT2;
  // reverse
  phi_r = sc[10];
  Kc = refCinv * exp(2.000000 * g_RT[3] - g_RT[10]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[3]) + (h_RT[10]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= 2 * q; // CH3
  wdot[10] += q;    // C2H6
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor;
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor + k_f * 2.000000 * sc[3];
  dqdc[4] = q_nocor;
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  dqdc[9] = q_nocor;
  dqdc[10] = q_nocor - k_r;
  dqdc[11] = q_nocor;
  for (int k = 0; k < 12; k++) {
    J[13 * k + 3] += -2 * dqdc[k];
    J[13 * k + 10] += dqdc[k];
  }
  J[159] += -2 * dqdT; // dwdot[CH3]/dT
  J[166] += dqdT;      // dwdot[C2H6]/dT

  // reaction 1: CH4 + H <=> CH3 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[4];
  k_f = 0.352683948427848 *
        exp(2.76711916768222 * logT - (8314.80590579313) * invT);
  dlnkfdT = 2.76711916768222 * invT + (8314.80590579313) * invT2;
  // reverse
  phi_r = sc[0] * sc[3];
  Kc = exp(-g_RT[0] + g_RT[1] - g_RT[3] + g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[4]) + (h_RT[0] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // H2
  wdot[1] -= q; // H
  wdot[3] += q; // CH3
  wdot[4] -= q; // CH4
  // d()/d[H2]
  dqdci = -k_r * sc[3];
  J[0] += dqdci; // dwdot[H2]/d[H2]
  J[1] -= dqdci; // dwdot[H]/d[H2]
  J[3] += dqdci; // dwdot[CH3]/d[H2]
  J[4] -= dqdci; // dwdot[CH4]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[4];
  J[13] += dqdci; // dwdot[H2]/d[H]
  J[14] -= dqdci; // dwdot[H]/d[H]
  J[16] += dqdci; // dwdot[CH3]/d[H]
  J[17] -= dqdci; // dwdot[CH4]/d[H]
  // d()/d[CH3]
  dqdci = -k_r * sc[0];
  J[39] += dqdci; // dwdot[H2]/d[CH3]
  J[40] -= dqdci; // dwdot[H]/d[CH3]
  J[42] += dqdci; // dwdot[CH3]/d[CH3]
  J[43] -= dqdci; // dwdot[CH4]/d[CH3]
  // d()/d[CH4]
  dqdci = +k_f * sc[1];
  J[52] += dqdci; // dwdot[H2]/d[CH4]
  J[53] -= dqdci; // dwdot[H]/d[CH4]
  J[55] += dqdci; // dwdot[CH3]/d[CH4]
  J[56] -= dqdci; // dwdot[CH4]/d[CH4]
  // d()/dT
  J[156] += dqdT; // dwdot[H2]/dT
  J[157] -= dqdT; // dwdot[H]/dT
  J[159] += dqdT; // dwdot[CH3]/dT
  J[160] -= dqdT; // dwdot[CH4]/dT

  // reaction 2: 2 CH3 <=> C2H5 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[3] * sc[3]);
  k_f = 36152.1969455548 *
        exp(0.845120206098996 * logT - (3909.67459913287) * invT);
  dlnkfdT = 0.845120206098996 * invT + (3909.67459913287) * invT2;
  // reverse
  phi_r = sc[1] * sc[9];
  Kc = exp(-g_RT[1] + 2.000000 * g_RT[3] - g_RT[9]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[3]) + (h_RT[1] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;     // H
  wdot[3] -= 2 * q; // CH3
  wdot[9] += q;     // C2H5
  // d()/d[H]
  dqdci = -k_r * sc[9];
  J[14] += dqdci;      // dwdot[H]/d[H]
  J[16] += -2 * dqdci; // dwdot[CH3]/d[H]
  J[22] += dqdci;      // dwdot[C2H5]/d[H]
  // d()/d[CH3]
  dqdci = +k_f * 2.000000 * sc[3];
  J[40] += dqdci;      // dwdot[H]/d[CH3]
  J[42] += -2 * dqdci; // dwdot[CH3]/d[CH3]
  J[48] += dqdci;      // dwdot[C2H5]/d[CH3]
  // d()/d[C2H5]
  dqdci = -k_r * sc[1];
  J[118] += dqdci;      // dwdot[H]/d[C2H5]
  J[120] += -2 * dqdci; // dwdot[CH3]/d[C2H5]
  J[126] += dqdci;      // dwdot[C2H5]/d[C2H5]
  // d()/dT
  J[157] += dqdT;      // dwdot[H]/dT
  J[159] += -2 * dqdT; // dwdot[CH3]/dT
  J[165] += dqdT;      // dwdot[C2H5]/dT

  // reaction 4: C2H4 + H <=> C2H3 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[8];
  k_f =
    2512.05632509109 * exp(1.53801468328155 * logT - (8383.73163879634) * invT);
  dlnkfdT = 1.53801468328155 * invT + (8383.73163879634) * invT2;
  // reverse
  phi_r = sc[0] * sc[7];
  Kc = exp(-g_RT[0] + g_RT[1] - g_RT[7] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[8]) + (h_RT[0] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // H2
  wdot[1] -= q; // H
  wdot[7] += q; // C2H3
  wdot[8] -= q; // C2H4
  // d()/d[H2]
  dqdci = -k_r * sc[7];
  J[0] += dqdci; // dwdot[H2]/d[H2]
  J[1] -= dqdci; // dwdot[H]/d[H2]
  J[7] += dqdci; // dwdot[C2H3]/d[H2]
  J[8] -= dqdci; // dwdot[C2H4]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[8];
  J[13] += dqdci; // dwdot[H2]/d[H]
  J[14] -= dqdci; // dwdot[H]/d[H]
  J[20] += dqdci; // dwdot[C2H3]/d[H]
  J[21] -= dqdci; // dwdot[C2H4]/d[H]
  // d()/d[C2H3]
  dqdci = -k_r * sc[0];
  J[91] += dqdci; // dwdot[H2]/d[C2H3]
  J[92] -= dqdci; // dwdot[H]/d[C2H3]
  J[98] += dqdci; // dwdot[C2H3]/d[C2H3]
  J[99] -= dqdci; // dwdot[C2H4]/d[C2H3]
  // d()/d[C2H4]
  dqdci = +k_f * sc[1];
  J[104] += dqdci; // dwdot[H2]/d[C2H4]
  J[105] -= dqdci; // dwdot[H]/d[C2H4]
  J[111] += dqdci; // dwdot[C2H3]/d[C2H4]
  J[112] -= dqdci; // dwdot[C2H4]/d[C2H4]
  // d()/dT
  J[156] += dqdT; // dwdot[H2]/dT
  J[157] -= dqdT; // dwdot[H]/dT
  J[163] += dqdT; // dwdot[C2H3]/dT
  J[164] -= dqdT; // dwdot[C2H4]/dT

  // reaction 7: C2H + H2 <=> C2H2 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[5];
  k_f =
    5372180.16519361 * exp(1.6080006615735 * logT - (15642.633813977) * invT);
  dlnkfdT = 1.6080006615735 * invT + (15642.633813977) * invT2;
  // reverse
  phi_r = sc[1] * sc[6];
  Kc = exp(g_RT[0] - g_RT[1] + g_RT[5] - g_RT[6]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[5]) + (h_RT[1] + h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // H2
  wdot[1] += q; // H
  wdot[5] -= q; // C2H
  wdot[6] += q; // C2H2
  // d()/d[H2]
  dqdci = +k_f * sc[5];
  J[0] -= dqdci; // dwdot[H2]/d[H2]
  J[1] += dqdci; // dwdot[H]/d[H2]
  J[5] -= dqdci; // dwdot[C2H]/d[H2]
  J[6] += dqdci; // dwdot[C2H2]/d[H2]
  // d()/d[H]
  dqdci = -k_r * sc[6];
  J[13] -= dqdci; // dwdot[H2]/d[H]
  J[14] += dqdci; // dwdot[H]/d[H]
  J[18] -= dqdci; // dwdot[C2H]/d[H]
  J[19] += dqdci; // dwdot[C2H2]/d[H]
  // d()/d[C2H]
  dqdci = +k_f * sc[0];
  J[65] -= dqdci; // dwdot[H2]/d[C2H]
  J[66] += dqdci; // dwdot[H]/d[C2H]
  J[70] -= dqdci; // dwdot[C2H]/d[C2H]
  J[71] += dqdci; // dwdot[C2H2]/d[C2H]
  // d()/d[C2H2]
  dqdci = -k_r * sc[1];
  J[78] -= dqdci; // dwdot[H2]/d[C2H2]
  J[79] += dqdci; // dwdot[H]/d[C2H2]
  J[83] -= dqdci; // dwdot[C2H]/d[C2H2]
  J[84] += dqdci; // dwdot[C2H2]/d[C2H2]
  // d()/dT
  J[156] -= dqdT; // dwdot[H2]/dT
  J[157] += dqdT; // dwdot[H]/dT
  J[161] -= dqdT; // dwdot[C2H]/dT
  J[162] += dqdT; // dwdot[C2H2]/dT

  // reaction 8: CH2 + H2 <=> CH3 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[2];
  k_f =
    100.17524517356 * exp(2.68606997955989 * logT - (5251.06261291261) * invT);
  dlnkfdT = 2.68606997955989 * invT + (5251.06261291261) * invT2;
  // reverse
  phi_r = sc[1] * sc[3];
  Kc = exp(g_RT[0] - g_RT[1] + g_RT[2] - g_RT[3]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[2]) + (h_RT[1] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // H2
  wdot[1] += q; // H
  wdot[2] -= q; // CH2
  wdot[3] += q; // CH3
  // d()/d[H2]
  dqdci = +k_f * sc[2];
  J[0] -= dqdci; // dwdot[H2]/d[H2]
  J[1] += dqdci; // dwdot[H]/d[H2]
  J[2] -= dqdci; // dwdot[CH2]/d[H2]
  J[3] += dqdci; // dwdot[CH3]/d[H2]
  // d()/d[H]
  dqdci = -k_r * sc[3];
  J[13] -= dqdci; // dwdot[H2]/d[H]
  J[14] += dqdci; // dwdot[H]/d[H]
  J[15] -= dqdci; // dwdot[CH2]/d[H]
  J[16] += dqdci; // dwdot[CH3]/d[H]
  // d()/d[CH2]
  dqdci = +k_f * sc[0];
  J[26] -= dqdci; // dwdot[H2]/d[CH2]
  J[27] += dqdci; // dwdot[H]/d[CH2]
  J[28] -= dqdci; // dwdot[CH2]/d[CH2]
  J[29] += dqdci; // dwdot[CH3]/d[CH2]
  // d()/d[CH3]
  dqdci = -k_r * sc[1];
  J[39] -= dqdci; // dwdot[H2]/d[CH3]
  J[40] += dqdci; // dwdot[H]/d[CH3]
  J[41] -= dqdci; // dwdot[CH2]/d[CH3]
  J[42] += dqdci; // dwdot[CH3]/d[CH3]
  // d()/dT
  J[156] -= dqdT; // dwdot[H2]/dT
  J[157] += dqdT; // dwdot[H]/dT
  J[158] -= dqdT; // dwdot[CH2]/dT
  J[159] += dqdT; // dwdot[CH3]/dT

  // reaction 9: CH2 + CH3 <=> C2H4 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[3];
  k_f = 4.0153594627338 *
        exp(0.575581019159047 * logT - (0.443945260758714) * invT);
  dlnkfdT = 0.575581019159047 * invT + (0.443945260758714) * invT2;
  // reverse
  phi_r = sc[1] * sc[8];
  Kc = exp(-g_RT[1] + g_RT[2] + g_RT[3] - g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[3]) + (h_RT[1] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // H
  wdot[2] -= q; // CH2
  wdot[3] -= q; // CH3
  wdot[8] += q; // C2H4
  // d()/d[H]
  dqdci = -k_r * sc[8];
  J[14] += dqdci; // dwdot[H]/d[H]
  J[15] -= dqdci; // dwdot[CH2]/d[H]
  J[16] -= dqdci; // dwdot[CH3]/d[H]
  J[21] += dqdci; // dwdot[C2H4]/d[H]
  // d()/d[CH2]
  dqdci = +k_f * sc[3];
  J[27] += dqdci; // dwdot[H]/d[CH2]
  J[28] -= dqdci; // dwdot[CH2]/d[CH2]
  J[29] -= dqdci; // dwdot[CH3]/d[CH2]
  J[34] += dqdci; // dwdot[C2H4]/d[CH2]
  // d()/d[CH3]
  dqdci = +k_f * sc[2];
  J[40] += dqdci; // dwdot[H]/d[CH3]
  J[41] -= dqdci; // dwdot[CH2]/d[CH3]
  J[42] -= dqdci; // dwdot[CH3]/d[CH3]
  J[47] += dqdci; // dwdot[C2H4]/d[CH3]
  // d()/d[C2H4]
  dqdci = -k_r * sc[1];
  J[105] += dqdci; // dwdot[H]/d[C2H4]
  J[106] -= dqdci; // dwdot[CH2]/d[C2H4]
  J[107] -= dqdci; // dwdot[CH3]/d[C2H4]
  J[112] += dqdci; // dwdot[C2H4]/d[C2H4]
  // d()/dT
  J[157] += dqdT; // dwdot[H]/dT
  J[158] -= dqdT; // dwdot[CH2]/dT
  J[159] -= dqdT; // dwdot[CH3]/dT
  J[164] += dqdT; // dwdot[C2H4]/dT

  // reaction 11: C2H2 <=> 2 C(s) + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6];
  k_f = 123.472588382411 *
        exp(-0.293562938580551 * logT - (5.10231730398989) * invT);
  dlnkfdT = -0.293562938580551 * invT + (5.10231730398989) * invT2;
  // reverse
  phi_r = sc[0] * (sc[11] * sc[11]);
  Kc = (refC * refC) * exp(-g_RT[0] + g_RT[6] - 2.000000 * g_RT[11]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6]) + (h_RT[0] + 2.000000 * h_RT[11]) - 2.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q;      // H2
  wdot[6] -= q;      // C2H2
  wdot[11] += 2 * q; // C(s)
  // d()/d[H2]
  dqdci = -k_r * sc[11] * sc[11];
  J[0] += dqdci;      // dwdot[H2]/d[H2]
  J[6] -= dqdci;      // dwdot[C2H2]/d[H2]
  J[11] += 2 * dqdci; // dwdot[C(s)]/d[H2]
  // d()/d[C2H2]
  dqdci = +k_f;
  J[78] += dqdci;     // dwdot[H2]/d[C2H2]
  J[84] -= dqdci;     // dwdot[C2H2]/d[C2H2]
  J[89] += 2 * dqdci; // dwdot[C(s)]/d[C2H2]
  // d()/d[C(s)]
  dqdci = -k_r * sc[0] * 2.000000 * sc[11];
  J[143] += dqdci;     // dwdot[H2]/d[C(s)]
  J[149] -= dqdci;     // dwdot[C2H2]/d[C(s)]
  J[154] += 2 * dqdci; // dwdot[C(s)]/d[C(s)]
  // d()/dT
  J[156] += dqdT;     // dwdot[H2]/dT
  J[162] -= dqdT;     // dwdot[C2H2]/dT
  J[167] += 2 * dqdT; // dwdot[C(s)]/dT

  // reaction 13: C2H6 + H <=> C2H5 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[10];
  k_f =
    2853873.1839057 * exp(1.43294743399092 * logT - (12431.3149961588) * invT);
  dlnkfdT = 1.43294743399092 * invT + (12431.3149961588) * invT2;
  // reverse
  phi_r = sc[0] * sc[9];
  Kc = exp(-g_RT[0] + g_RT[1] - g_RT[9] + g_RT[10]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[10]) + (h_RT[0] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q;  // H2
  wdot[1] -= q;  // H
  wdot[9] += q;  // C2H5
  wdot[10] -= q; // C2H6
  // d()/d[H2]
  dqdci = -k_r * sc[9];
  J[0] += dqdci;  // dwdot[H2]/d[H2]
  J[1] -= dqdci;  // dwdot[H]/d[H2]
  J[9] += dqdci;  // dwdot[C2H5]/d[H2]
  J[10] -= dqdci; // dwdot[C2H6]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[10];
  J[13] += dqdci; // dwdot[H2]/d[H]
  J[14] -= dqdci; // dwdot[H]/d[H]
  J[22] += dqdci; // dwdot[C2H5]/d[H]
  J[23] -= dqdci; // dwdot[C2H6]/d[H]
  // d()/d[C2H5]
  dqdci = -k_r * sc[0];
  J[117] += dqdci; // dwdot[H2]/d[C2H5]
  J[118] -= dqdci; // dwdot[H]/d[C2H5]
  J[126] += dqdci; // dwdot[C2H5]/d[C2H5]
  J[127] -= dqdci; // dwdot[C2H6]/d[C2H5]
  // d()/d[C2H6]
  dqdci = +k_f * sc[1];
  J[130] += dqdci; // dwdot[H2]/d[C2H6]
  J[131] -= dqdci; // dwdot[H]/d[C2H6]
  J[139] += dqdci; // dwdot[C2H5]/d[C2H6]
  J[140] -= dqdci; // dwdot[C2H6]/d[C2H6]
  // d()/dT
  J[156] += dqdT; // dwdot[H2]/dT
  J[157] -= dqdT; // dwdot[H]/dT
  J[165] += dqdT; // dwdot[C2H5]/dT
  J[166] -= dqdT; // dwdot[C2H6]/dT

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
  // reaction 0: CH3 + H + M <=> CH4 + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[1] * sc[3];
  k_f = 18529779629.8165 *
        exp(-1.19245049206076 * logT - (85.2283723825864) * invT);
  dlnkfdT = -1.19245049206076 * invT + (85.2283723825864) * invT2;
  // reverse
  phi_r = sc[4];
  Kc = refCinv * exp(g_RT[1] + g_RT[3] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[3]) + (h_RT[4]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // H
  wdot[3] -= q; // CH3
  wdot[4] += q; // CH4
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[H]
    dqdci = +k_f * sc[3];
    J[14] -= dqdci; // dwdot[H]/d[H]
    J[16] -= dqdci; // dwdot[CH3]/d[H]
    J[17] += dqdci; // dwdot[CH4]/d[H]
    // d()/d[CH3]
    dqdci = +k_f * sc[1];
    J[40] -= dqdci; // dwdot[H]/d[CH3]
    J[42] -= dqdci; // dwdot[CH3]/d[CH3]
    J[43] += dqdci; // dwdot[CH4]/d[CH3]
    // d()/d[CH4]
    dqdci = -k_r;
    J[53] -= dqdci; // dwdot[H]/d[CH4]
    J[55] -= dqdci; // dwdot[CH3]/d[CH4]
    J[56] += dqdci; // dwdot[CH4]/d[CH4]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor + k_f * sc[3];
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor + k_f * sc[1];
    dqdc[4] = q_nocor - k_r;
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    dqdc[9] = q_nocor;
    dqdc[10] = q_nocor;
    dqdc[11] = q_nocor;
    for (int k = 0; k < 12; k++) {
      J[13 * k + 1] -= dqdc[k];
      J[13 * k + 3] -= dqdc[k];
      J[13 * k + 4] += dqdc[k];
    }
  }
  J[157] -= dqdT; // dwdot[H]/dT
  J[159] -= dqdT; // dwdot[CH3]/dT
  J[160] += dqdT; // dwdot[CH4]/dT

  // reaction 3: C2H4 + H (+M) <=> C2H5 (+M)
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[1] * sc[8];
  k_f =
    9867.16302841882 * exp(1.00304123676575 * logT - (1297.74828748432) * invT);
  dlnkfdT = 1.00304123676575 * invT + (1297.74828748432) * invT2;
  // reverse
  phi_r = sc[9];
  Kc = refCinv * exp(g_RT[1] + g_RT[8] - g_RT[9]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[8]) + (h_RT[9]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // H
  wdot[8] -= q; // C2H4
  wdot[9] += q; // C2H5
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[H]
    dqdci = +k_f * sc[8];
    J[14] -= dqdci; // dwdot[H]/d[H]
    J[21] -= dqdci; // dwdot[C2H4]/d[H]
    J[22] += dqdci; // dwdot[C2H5]/d[H]
    // d()/d[C2H4]
    dqdci = +k_f * sc[1];
    J[105] -= dqdci; // dwdot[H]/d[C2H4]
    J[112] -= dqdci; // dwdot[C2H4]/d[C2H4]
    J[113] += dqdci; // dwdot[C2H5]/d[C2H4]
    // d()/d[C2H5]
    dqdci = -k_r;
    J[118] -= dqdci; // dwdot[H]/d[C2H5]
    J[125] -= dqdci; // dwdot[C2H4]/d[C2H5]
    J[126] += dqdci; // dwdot[C2H5]/d[C2H5]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor + k_f * sc[8];
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor;
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor + k_f * sc[1];
    dqdc[9] = q_nocor - k_r;
    dqdc[10] = q_nocor;
    dqdc[11] = q_nocor;
    for (int k = 0; k < 12; k++) {
      J[13 * k + 1] -= dqdc[k];
      J[13 * k + 8] -= dqdc[k];
      J[13 * k + 9] += dqdc[k];
    }
  }
  J[157] -= dqdT; // dwdot[H]/dT
  J[164] -= dqdT; // dwdot[C2H4]/dT
  J[165] += dqdT; // dwdot[C2H5]/dT

  // reaction 5: C2H2 + H (+M) <=> C2H3 (+M)
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[1] * sc[6];
  k_f = 68388.8864690645 *
        exp(-0.86014726711066 * logT - (1267.69380866152) * invT);
  dlnkfdT = -0.86014726711066 * invT + (1267.69380866152) * invT2;
  // reverse
  phi_r = sc[7];
  Kc = refCinv * exp(g_RT[1] + g_RT[6] - g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[6]) + (h_RT[7]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // H
  wdot[6] -= q; // C2H2
  wdot[7] += q; // C2H3
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[H]
    dqdci = +k_f * sc[6];
    J[14] -= dqdci; // dwdot[H]/d[H]
    J[19] -= dqdci; // dwdot[C2H2]/d[H]
    J[20] += dqdci; // dwdot[C2H3]/d[H]
    // d()/d[C2H2]
    dqdci = +k_f * sc[1];
    J[79] -= dqdci; // dwdot[H]/d[C2H2]
    J[84] -= dqdci; // dwdot[C2H2]/d[C2H2]
    J[85] += dqdci; // dwdot[C2H3]/d[C2H2]
    // d()/d[C2H3]
    dqdci = -k_r;
    J[92] -= dqdci; // dwdot[H]/d[C2H3]
    J[97] -= dqdci; // dwdot[C2H2]/d[C2H3]
    J[98] += dqdci; // dwdot[C2H3]/d[C2H3]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor + k_f * sc[6];
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor;
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor + k_f * sc[1];
    dqdc[7] = q_nocor - k_r;
    dqdc[8] = q_nocor;
    dqdc[9] = q_nocor;
    dqdc[10] = q_nocor;
    dqdc[11] = q_nocor;
    for (int k = 0; k < 12; k++) {
      J[13 * k + 1] -= dqdc[k];
      J[13 * k + 6] -= dqdc[k];
      J[13 * k + 7] += dqdc[k];
    }
  }
  J[157] -= dqdT; // dwdot[H]/dT
  J[162] -= dqdT; // dwdot[C2H2]/dT
  J[163] += dqdT; // dwdot[C2H3]/dT

  // reaction 6: C2H + H (+M) <=> C2H2 (+M)
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[1] * sc[5];
  k_f = 4668349273.45473 *
        exp(-0.803167128908363 * logT - (0.00158333685917708) * invT);
  dlnkfdT = -0.803167128908363 * invT + (0.00158333685917708) * invT2;
  // reverse
  phi_r = sc[6];
  Kc = refCinv * exp(g_RT[1] + g_RT[5] - g_RT[6]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[5]) + (h_RT[6]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // H
  wdot[5] -= q; // C2H
  wdot[6] += q; // C2H2
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[H]
    dqdci = +k_f * sc[5];
    J[14] -= dqdci; // dwdot[H]/d[H]
    J[18] -= dqdci; // dwdot[C2H]/d[H]
    J[19] += dqdci; // dwdot[C2H2]/d[H]
    // d()/d[C2H]
    dqdci = +k_f * sc[1];
    J[66] -= dqdci; // dwdot[H]/d[C2H]
    J[70] -= dqdci; // dwdot[C2H]/d[C2H]
    J[71] += dqdci; // dwdot[C2H2]/d[C2H]
    // d()/d[C2H2]
    dqdci = -k_r;
    J[79] -= dqdci; // dwdot[H]/d[C2H2]
    J[83] -= dqdci; // dwdot[C2H]/d[C2H2]
    J[84] += dqdci; // dwdot[C2H2]/d[C2H2]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor + k_f * sc[5];
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor;
    dqdc[5] = q_nocor + k_f * sc[1];
    dqdc[6] = q_nocor - k_r;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    dqdc[9] = q_nocor;
    dqdc[10] = q_nocor;
    dqdc[11] = q_nocor;
    for (int k = 0; k < 12; k++) {
      J[13 * k + 1] -= dqdc[k];
      J[13 * k + 5] -= dqdc[k];
      J[13 * k + 6] += dqdc[k];
    }
  }
  J[157] -= dqdT; // dwdot[H]/dT
  J[161] -= dqdT; // dwdot[C2H]/dT
  J[162] += dqdT; // dwdot[C2H2]/dT

  // reaction 10: C2H4 + M <=> C2H2 + H2 + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[8];
  k_f = 49191261.0996573 *
        exp(0.794734252072147 * logT - (31648.5563093388) * invT);
  dlnkfdT = 0.794734252072147 * invT + (31648.5563093388) * invT2;
  // reverse
  phi_r = sc[0] * sc[6];
  Kc = refC * exp(-g_RT[0] - g_RT[6] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8]) + (h_RT[0] + h_RT[6]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // H2
  wdot[6] += q; // C2H2
  wdot[8] -= q; // C2H4
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[H2]
    dqdci = -k_r * sc[6];
    J[0] += dqdci; // dwdot[H2]/d[H2]
    J[6] += dqdci; // dwdot[C2H2]/d[H2]
    J[8] -= dqdci; // dwdot[C2H4]/d[H2]
    // d()/d[C2H2]
    dqdci = -k_r * sc[0];
    J[78] += dqdci; // dwdot[H2]/d[C2H2]
    J[84] += dqdci; // dwdot[C2H2]/d[C2H2]
    J[86] -= dqdci; // dwdot[C2H4]/d[C2H2]
    // d()/d[C2H4]
    dqdci = +k_f;
    J[104] += dqdci; // dwdot[H2]/d[C2H4]
    J[110] += dqdci; // dwdot[C2H2]/d[C2H4]
    J[112] -= dqdci; // dwdot[C2H4]/d[C2H4]
  } else {
    dqdc[0] = q_nocor - k_r * sc[6];
    dqdc[1] = q_nocor;
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor;
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor - k_r * sc[0];
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor + k_f;
    dqdc[9] = q_nocor;
    dqdc[10] = q_nocor;
    dqdc[11] = q_nocor;
    for (int k = 0; k < 12; k++) {
      J[13 * k + 0] += dqdc[k];
      J[13 * k + 6] += dqdc[k];
      J[13 * k + 8] -= dqdc[k];
    }
  }
  J[156] += dqdT; // dwdot[H2]/dT
  J[162] += dqdT; // dwdot[C2H2]/dT
  J[164] -= dqdT; // dwdot[C2H4]/dT

  // reaction 12: 2 CH3 + M <=> C2H6 + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = (sc[3] * sc[3]);
  k_f = 1026575730779.29 *
        exp(-0.592670538495204 * logT - (642.311112548238) * invT);
  dlnkfdT = -0.592670538495204 * invT + (642.311112548238) * invT2;
  // reverse
  phi_r = sc[10];
  Kc = refCinv * exp(2.000000 * g_RT[3] - g_RT[10]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[3]) + (h_RT[10]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= 2 * q; // CH3
  wdot[10] += q;    // C2H6
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[CH3]
    dqdci = +k_f * 2.000000 * sc[3];
    J[42] += -2 * dqdci; // dwdot[CH3]/d[CH3]
    J[49] += dqdci;      // dwdot[C2H6]/d[CH3]
    // d()/d[C2H6]
    dqdci = -k_r;
    J[133] += -2 * dqdci; // dwdot[CH3]/d[C2H6]
    J[140] += dqdci;      // dwdot[C2H6]/d[C2H6]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor;
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor + k_f * 2.000000 * sc[3];
    dqdc[4] = q_nocor;
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    dqdc[9] = q_nocor;
    dqdc[10] = q_nocor - k_r;
    dqdc[11] = q_nocor;
    for (int k = 0; k < 12; k++) {
      J[13 * k + 3] += -2 * dqdc[k];
      J[13 * k + 10] += dqdc[k];
    }
  }
  J[159] += -2 * dqdT; // dwdot[CH3]/dT
  J[166] += dqdT;      // dwdot[C2H6]/dT

  // reaction 1: CH4 + H <=> CH3 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[4];
  k_f = 0.352683948427848 *
        exp(2.76711916768222 * logT - (8314.80590579313) * invT);
  dlnkfdT = 2.76711916768222 * invT + (8314.80590579313) * invT2;
  // reverse
  phi_r = sc[0] * sc[3];
  Kc = exp(-g_RT[0] + g_RT[1] - g_RT[3] + g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[4]) + (h_RT[0] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // H2
  wdot[1] -= q; // H
  wdot[3] += q; // CH3
  wdot[4] -= q; // CH4
  // d()/d[H2]
  dqdci = -k_r * sc[3];
  J[0] += dqdci; // dwdot[H2]/d[H2]
  J[1] -= dqdci; // dwdot[H]/d[H2]
  J[3] += dqdci; // dwdot[CH3]/d[H2]
  J[4] -= dqdci; // dwdot[CH4]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[4];
  J[13] += dqdci; // dwdot[H2]/d[H]
  J[14] -= dqdci; // dwdot[H]/d[H]
  J[16] += dqdci; // dwdot[CH3]/d[H]
  J[17] -= dqdci; // dwdot[CH4]/d[H]
  // d()/d[CH3]
  dqdci = -k_r * sc[0];
  J[39] += dqdci; // dwdot[H2]/d[CH3]
  J[40] -= dqdci; // dwdot[H]/d[CH3]
  J[42] += dqdci; // dwdot[CH3]/d[CH3]
  J[43] -= dqdci; // dwdot[CH4]/d[CH3]
  // d()/d[CH4]
  dqdci = +k_f * sc[1];
  J[52] += dqdci; // dwdot[H2]/d[CH4]
  J[53] -= dqdci; // dwdot[H]/d[CH4]
  J[55] += dqdci; // dwdot[CH3]/d[CH4]
  J[56] -= dqdci; // dwdot[CH4]/d[CH4]
  // d()/dT
  J[156] += dqdT; // dwdot[H2]/dT
  J[157] -= dqdT; // dwdot[H]/dT
  J[159] += dqdT; // dwdot[CH3]/dT
  J[160] -= dqdT; // dwdot[CH4]/dT

  // reaction 2: 2 CH3 <=> C2H5 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[3] * sc[3]);
  k_f = 36152.1969455548 *
        exp(0.845120206098996 * logT - (3909.67459913287) * invT);
  dlnkfdT = 0.845120206098996 * invT + (3909.67459913287) * invT2;
  // reverse
  phi_r = sc[1] * sc[9];
  Kc = exp(-g_RT[1] + 2.000000 * g_RT[3] - g_RT[9]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[3]) + (h_RT[1] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;     // H
  wdot[3] -= 2 * q; // CH3
  wdot[9] += q;     // C2H5
  // d()/d[H]
  dqdci = -k_r * sc[9];
  J[14] += dqdci;      // dwdot[H]/d[H]
  J[16] += -2 * dqdci; // dwdot[CH3]/d[H]
  J[22] += dqdci;      // dwdot[C2H5]/d[H]
  // d()/d[CH3]
  dqdci = +k_f * 2.000000 * sc[3];
  J[40] += dqdci;      // dwdot[H]/d[CH3]
  J[42] += -2 * dqdci; // dwdot[CH3]/d[CH3]
  J[48] += dqdci;      // dwdot[C2H5]/d[CH3]
  // d()/d[C2H5]
  dqdci = -k_r * sc[1];
  J[118] += dqdci;      // dwdot[H]/d[C2H5]
  J[120] += -2 * dqdci; // dwdot[CH3]/d[C2H5]
  J[126] += dqdci;      // dwdot[C2H5]/d[C2H5]
  // d()/dT
  J[157] += dqdT;      // dwdot[H]/dT
  J[159] += -2 * dqdT; // dwdot[CH3]/dT
  J[165] += dqdT;      // dwdot[C2H5]/dT

  // reaction 4: C2H4 + H <=> C2H3 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[8];
  k_f =
    2512.05632509109 * exp(1.53801468328155 * logT - (8383.73163879634) * invT);
  dlnkfdT = 1.53801468328155 * invT + (8383.73163879634) * invT2;
  // reverse
  phi_r = sc[0] * sc[7];
  Kc = exp(-g_RT[0] + g_RT[1] - g_RT[7] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[8]) + (h_RT[0] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // H2
  wdot[1] -= q; // H
  wdot[7] += q; // C2H3
  wdot[8] -= q; // C2H4
  // d()/d[H2]
  dqdci = -k_r * sc[7];
  J[0] += dqdci; // dwdot[H2]/d[H2]
  J[1] -= dqdci; // dwdot[H]/d[H2]
  J[7] += dqdci; // dwdot[C2H3]/d[H2]
  J[8] -= dqdci; // dwdot[C2H4]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[8];
  J[13] += dqdci; // dwdot[H2]/d[H]
  J[14] -= dqdci; // dwdot[H]/d[H]
  J[20] += dqdci; // dwdot[C2H3]/d[H]
  J[21] -= dqdci; // dwdot[C2H4]/d[H]
  // d()/d[C2H3]
  dqdci = -k_r * sc[0];
  J[91] += dqdci; // dwdot[H2]/d[C2H3]
  J[92] -= dqdci; // dwdot[H]/d[C2H3]
  J[98] += dqdci; // dwdot[C2H3]/d[C2H3]
  J[99] -= dqdci; // dwdot[C2H4]/d[C2H3]
  // d()/d[C2H4]
  dqdci = +k_f * sc[1];
  J[104] += dqdci; // dwdot[H2]/d[C2H4]
  J[105] -= dqdci; // dwdot[H]/d[C2H4]
  J[111] += dqdci; // dwdot[C2H3]/d[C2H4]
  J[112] -= dqdci; // dwdot[C2H4]/d[C2H4]
  // d()/dT
  J[156] += dqdT; // dwdot[H2]/dT
  J[157] -= dqdT; // dwdot[H]/dT
  J[163] += dqdT; // dwdot[C2H3]/dT
  J[164] -= dqdT; // dwdot[C2H4]/dT

  // reaction 7: C2H + H2 <=> C2H2 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[5];
  k_f =
    5372180.16519361 * exp(1.6080006615735 * logT - (15642.633813977) * invT);
  dlnkfdT = 1.6080006615735 * invT + (15642.633813977) * invT2;
  // reverse
  phi_r = sc[1] * sc[6];
  Kc = exp(g_RT[0] - g_RT[1] + g_RT[5] - g_RT[6]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[5]) + (h_RT[1] + h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // H2
  wdot[1] += q; // H
  wdot[5] -= q; // C2H
  wdot[6] += q; // C2H2
  // d()/d[H2]
  dqdci = +k_f * sc[5];
  J[0] -= dqdci; // dwdot[H2]/d[H2]
  J[1] += dqdci; // dwdot[H]/d[H2]
  J[5] -= dqdci; // dwdot[C2H]/d[H2]
  J[6] += dqdci; // dwdot[C2H2]/d[H2]
  // d()/d[H]
  dqdci = -k_r * sc[6];
  J[13] -= dqdci; // dwdot[H2]/d[H]
  J[14] += dqdci; // dwdot[H]/d[H]
  J[18] -= dqdci; // dwdot[C2H]/d[H]
  J[19] += dqdci; // dwdot[C2H2]/d[H]
  // d()/d[C2H]
  dqdci = +k_f * sc[0];
  J[65] -= dqdci; // dwdot[H2]/d[C2H]
  J[66] += dqdci; // dwdot[H]/d[C2H]
  J[70] -= dqdci; // dwdot[C2H]/d[C2H]
  J[71] += dqdci; // dwdot[C2H2]/d[C2H]
  // d()/d[C2H2]
  dqdci = -k_r * sc[1];
  J[78] -= dqdci; // dwdot[H2]/d[C2H2]
  J[79] += dqdci; // dwdot[H]/d[C2H2]
  J[83] -= dqdci; // dwdot[C2H]/d[C2H2]
  J[84] += dqdci; // dwdot[C2H2]/d[C2H2]
  // d()/dT
  J[156] -= dqdT; // dwdot[H2]/dT
  J[157] += dqdT; // dwdot[H]/dT
  J[161] -= dqdT; // dwdot[C2H]/dT
  J[162] += dqdT; // dwdot[C2H2]/dT

  // reaction 8: CH2 + H2 <=> CH3 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[2];
  k_f =
    100.17524517356 * exp(2.68606997955989 * logT - (5251.06261291261) * invT);
  dlnkfdT = 2.68606997955989 * invT + (5251.06261291261) * invT2;
  // reverse
  phi_r = sc[1] * sc[3];
  Kc = exp(g_RT[0] - g_RT[1] + g_RT[2] - g_RT[3]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[2]) + (h_RT[1] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // H2
  wdot[1] += q; // H
  wdot[2] -= q; // CH2
  wdot[3] += q; // CH3
  // d()/d[H2]
  dqdci = +k_f * sc[2];
  J[0] -= dqdci; // dwdot[H2]/d[H2]
  J[1] += dqdci; // dwdot[H]/d[H2]
  J[2] -= dqdci; // dwdot[CH2]/d[H2]
  J[3] += dqdci; // dwdot[CH3]/d[H2]
  // d()/d[H]
  dqdci = -k_r * sc[3];
  J[13] -= dqdci; // dwdot[H2]/d[H]
  J[14] += dqdci; // dwdot[H]/d[H]
  J[15] -= dqdci; // dwdot[CH2]/d[H]
  J[16] += dqdci; // dwdot[CH3]/d[H]
  // d()/d[CH2]
  dqdci = +k_f * sc[0];
  J[26] -= dqdci; // dwdot[H2]/d[CH2]
  J[27] += dqdci; // dwdot[H]/d[CH2]
  J[28] -= dqdci; // dwdot[CH2]/d[CH2]
  J[29] += dqdci; // dwdot[CH3]/d[CH2]
  // d()/d[CH3]
  dqdci = -k_r * sc[1];
  J[39] -= dqdci; // dwdot[H2]/d[CH3]
  J[40] += dqdci; // dwdot[H]/d[CH3]
  J[41] -= dqdci; // dwdot[CH2]/d[CH3]
  J[42] += dqdci; // dwdot[CH3]/d[CH3]
  // d()/dT
  J[156] -= dqdT; // dwdot[H2]/dT
  J[157] += dqdT; // dwdot[H]/dT
  J[158] -= dqdT; // dwdot[CH2]/dT
  J[159] += dqdT; // dwdot[CH3]/dT

  // reaction 9: CH2 + CH3 <=> C2H4 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[3];
  k_f = 4.0153594627338 *
        exp(0.575581019159047 * logT - (0.443945260758714) * invT);
  dlnkfdT = 0.575581019159047 * invT + (0.443945260758714) * invT2;
  // reverse
  phi_r = sc[1] * sc[8];
  Kc = exp(-g_RT[1] + g_RT[2] + g_RT[3] - g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[3]) + (h_RT[1] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // H
  wdot[2] -= q; // CH2
  wdot[3] -= q; // CH3
  wdot[8] += q; // C2H4
  // d()/d[H]
  dqdci = -k_r * sc[8];
  J[14] += dqdci; // dwdot[H]/d[H]
  J[15] -= dqdci; // dwdot[CH2]/d[H]
  J[16] -= dqdci; // dwdot[CH3]/d[H]
  J[21] += dqdci; // dwdot[C2H4]/d[H]
  // d()/d[CH2]
  dqdci = +k_f * sc[3];
  J[27] += dqdci; // dwdot[H]/d[CH2]
  J[28] -= dqdci; // dwdot[CH2]/d[CH2]
  J[29] -= dqdci; // dwdot[CH3]/d[CH2]
  J[34] += dqdci; // dwdot[C2H4]/d[CH2]
  // d()/d[CH3]
  dqdci = +k_f * sc[2];
  J[40] += dqdci; // dwdot[H]/d[CH3]
  J[41] -= dqdci; // dwdot[CH2]/d[CH3]
  J[42] -= dqdci; // dwdot[CH3]/d[CH3]
  J[47] += dqdci; // dwdot[C2H4]/d[CH3]
  // d()/d[C2H4]
  dqdci = -k_r * sc[1];
  J[105] += dqdci; // dwdot[H]/d[C2H4]
  J[106] -= dqdci; // dwdot[CH2]/d[C2H4]
  J[107] -= dqdci; // dwdot[CH3]/d[C2H4]
  J[112] += dqdci; // dwdot[C2H4]/d[C2H4]
  // d()/dT
  J[157] += dqdT; // dwdot[H]/dT
  J[158] -= dqdT; // dwdot[CH2]/dT
  J[159] -= dqdT; // dwdot[CH3]/dT
  J[164] += dqdT; // dwdot[C2H4]/dT

  // reaction 11: C2H2 <=> 2 C(s) + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6];
  k_f = 123.472588382411 *
        exp(-0.293562938580551 * logT - (5.10231730398989) * invT);
  dlnkfdT = -0.293562938580551 * invT + (5.10231730398989) * invT2;
  // reverse
  phi_r = sc[0] * (sc[11] * sc[11]);
  Kc = (refC * refC) * exp(-g_RT[0] + g_RT[6] - 2.000000 * g_RT[11]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6]) + (h_RT[0] + 2.000000 * h_RT[11]) - 2.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q;      // H2
  wdot[6] -= q;      // C2H2
  wdot[11] += 2 * q; // C(s)
  // d()/d[H2]
  dqdci = -k_r * sc[11] * sc[11];
  J[0] += dqdci;      // dwdot[H2]/d[H2]
  J[6] -= dqdci;      // dwdot[C2H2]/d[H2]
  J[11] += 2 * dqdci; // dwdot[C(s)]/d[H2]
  // d()/d[C2H2]
  dqdci = +k_f;
  J[78] += dqdci;     // dwdot[H2]/d[C2H2]
  J[84] -= dqdci;     // dwdot[C2H2]/d[C2H2]
  J[89] += 2 * dqdci; // dwdot[C(s)]/d[C2H2]
  // d()/d[C(s)]
  dqdci = -k_r * sc[0] * 2.000000 * sc[11];
  J[143] += dqdci;     // dwdot[H2]/d[C(s)]
  J[149] -= dqdci;     // dwdot[C2H2]/d[C(s)]
  J[154] += 2 * dqdci; // dwdot[C(s)]/d[C(s)]
  // d()/dT
  J[156] += dqdT;     // dwdot[H2]/dT
  J[162] -= dqdT;     // dwdot[C2H2]/dT
  J[167] += 2 * dqdT; // dwdot[C(s)]/dT

  // reaction 13: C2H6 + H <=> C2H5 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[10];
  k_f =
    2853873.1839057 * exp(1.43294743399092 * logT - (12431.3149961588) * invT);
  dlnkfdT = 1.43294743399092 * invT + (12431.3149961588) * invT2;
  // reverse
  phi_r = sc[0] * sc[9];
  Kc = exp(-g_RT[0] + g_RT[1] - g_RT[9] + g_RT[10]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[10]) + (h_RT[0] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q;  // H2
  wdot[1] -= q;  // H
  wdot[9] += q;  // C2H5
  wdot[10] -= q; // C2H6
  // d()/d[H2]
  dqdci = -k_r * sc[9];
  J[0] += dqdci;  // dwdot[H2]/d[H2]
  J[1] -= dqdci;  // dwdot[H]/d[H2]
  J[9] += dqdci;  // dwdot[C2H5]/d[H2]
  J[10] -= dqdci; // dwdot[C2H6]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[10];
  J[13] += dqdci; // dwdot[H2]/d[H]
  J[14] -= dqdci; // dwdot[H]/d[H]
  J[22] += dqdci; // dwdot[C2H5]/d[H]
  J[23] -= dqdci; // dwdot[C2H6]/d[H]
  // d()/d[C2H5]
  dqdci = -k_r * sc[0];
  J[117] += dqdci; // dwdot[H2]/d[C2H5]
  J[118] -= dqdci; // dwdot[H]/d[C2H5]
  J[126] += dqdci; // dwdot[C2H5]/d[C2H5]
  J[127] -= dqdci; // dwdot[C2H6]/d[C2H5]
  // d()/d[C2H6]
  dqdci = +k_f * sc[1];
  J[130] += dqdci; // dwdot[H2]/d[C2H6]
  J[131] -= dqdci; // dwdot[H]/d[C2H6]
  J[139] += dqdci; // dwdot[C2H5]/d[C2H6]
  J[140] -= dqdci; // dwdot[C2H6]/d[C2H6]
  // d()/dT
  J[156] += dqdT; // dwdot[H2]/dT
  J[157] -= dqdT; // dwdot[H]/dT
  J[165] += dqdT; // dwdot[C2H5]/dT
  J[166] -= dqdT; // dwdot[C2H6]/dT

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
  WT[0] = 2.01600000E+00;
  WT[1] = 1.00800000E+00;
  WT[2] = 1.40270000E+01;
  WT[3] = 1.50350000E+01;
  WT[4] = 1.60430000E+01;
  WT[5] = 2.50300000E+01;
  WT[6] = 2.60380000E+01;
  WT[7] = 2.70460000E+01;
  WT[8] = 2.80540000E+01;
  WT[9] = 2.90620000E+01;
  WT[10] = 3.00700000E+01;
  WT[11] = 1.20110000E+01;
}

// the lennard-jones potential well depth eps/kb in K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetEPS(amrex::Real* EPS)
{
  EPS[0] = 3.80000000E+01;
  EPS[1] = 1.45000000E+02;
  EPS[2] = 1.44000000E+02;
  EPS[3] = 1.44000000E+02;
  EPS[4] = 1.41400000E+02;
  EPS[5] = 2.09000000E+02;
  EPS[6] = 2.09000000E+02;
  EPS[7] = 2.09000000E+02;
  EPS[8] = 2.80800000E+02;
  EPS[9] = 2.52300000E+02;
  EPS[10] = 2.52300000E+02;
  EPS[11] = 7.14000000E+01;
}

// the lennard-jones collision diameter in Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetSIG(amrex::Real* SIG)
{
  SIG[0] = 2.92000000E+00;
  SIG[1] = 2.05000000E+00;
  SIG[2] = 3.80000000E+00;
  SIG[3] = 3.80000000E+00;
  SIG[4] = 3.74600000E+00;
  SIG[5] = 4.10000000E+00;
  SIG[6] = 4.10000000E+00;
  SIG[7] = 4.10000000E+00;
  SIG[8] = 3.97100000E+00;
  SIG[9] = 4.30200000E+00;
  SIG[10] = 4.30200000E+00;
  SIG[11] = 3.29800000E+00;
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
  POL[0] = 7.90000000E-01;
  POL[1] = 0.00000000E+00;
  POL[2] = 0.00000000E+00;
  POL[3] = 0.00000000E+00;
  POL[4] = 2.60000000E+00;
  POL[5] = 0.00000000E+00;
  POL[6] = 0.00000000E+00;
  POL[7] = 0.00000000E+00;
  POL[8] = 0.00000000E+00;
  POL[9] = 0.00000000E+00;
  POL[10] = 0.00000000E+00;
  POL[11] = 0.00000000E+00;
}

// the rotational relaxation collision number at 298 K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetZROT(amrex::Real* ZROT)
{
  ZROT[0] = 2.80000000E+02;
  ZROT[1] = 0.00000000E+00;
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
  NLIN[0] = 1;
  NLIN[1] = 0;
  NLIN[2] = 1;
  NLIN[3] = 1;
  NLIN[4] = 2;
  NLIN[5] = 1;
  NLIN[6] = 1;
  NLIN[7] = 2;
  NLIN[8] = 2;
  NLIN[9] = 2;
  NLIN[10] = 2;
  NLIN[11] = 0;
}

// Poly fits for the viscosities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFETA(amrex::Real* COFETA)
{
  COFETA[0] = -1.40419383E+01;
  COFETA[1] = 1.08789225E+00;
  COFETA[2] = -6.18592115E-02;
  COFETA[3] = 2.86838304E-03;
  COFETA[4] = -2.08449472E+01;
  COFETA[5] = 3.82794146E+00;
  COFETA[6] = -4.20999003E-01;
  COFETA[7] = 1.85596636E-02;
  COFETA[8] = -2.06998983E+01;
  COFETA[9] = 3.80445652E+00;
  COFETA[10] = -4.18022689E-01;
  COFETA[11] = 1.84337719E-02;
  COFETA[12] = -2.06651999E+01;
  COFETA[13] = 3.80445652E+00;
  COFETA[14] = -4.18022689E-01;
  COFETA[15] = 1.84337719E-02;
  COFETA[16] = -2.04642228E+01;
  COFETA[17] = 3.75363357E+00;
  COFETA[18] = -4.11756834E-01;
  COFETA[19] = 1.81766117E-02;
  COFETA[20] = -2.28885579E+01;
  COFETA[21] = 4.58175030E+00;
  COFETA[22] = -5.05624274E-01;
  COFETA[23] = 2.16754140E-02;
  COFETA[24] = -2.28688170E+01;
  COFETA[25] = 4.58175030E+00;
  COFETA[26] = -5.05624274E-01;
  COFETA[27] = 2.16754140E-02;
  COFETA[28] = -2.28498259E+01;
  COFETA[29] = 4.58175030E+00;
  COFETA[30] = -5.05624274E-01;
  COFETA[31] = 2.16754140E-02;
  COFETA[32] = -2.29541132E+01;
  COFETA[33] = 4.43807276E+00;
  COFETA[34] = -4.62324516E-01;
  COFETA[35] = 1.87855972E-02;
  COFETA[36] = -2.32762144E+01;
  COFETA[37] = 4.59930831E+00;
  COFETA[38] = -4.93738512E-01;
  COFETA[39] = 2.05755211E-02;
  COFETA[40] = -2.32591662E+01;
  COFETA[41] = 4.59930831E+00;
  COFETA[42] = -4.93738512E-01;
  COFETA[43] = 2.05755211E-02;
  COFETA[44] = -1.56481871E+01;
  COFETA[45] = 1.96214031E+00;
  COFETA[46] = -1.79397210E-01;
  COFETA[47] = 8.12672879E-03;
}

// Poly fits for the conductivities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFLAM(amrex::Real* COFLAM)
{
  COFLAM[0] = 4.34727634E+00;
  COFLAM[1] = 1.55347640E+00;
  COFLAM[2] = -1.60615528E-01;
  COFLAM[3] = 9.89934243E-03;
  COFLAM[4] = -1.29506647E+00;
  COFLAM[5] = 3.82794146E+00;
  COFLAM[6] = -4.20999003E-01;
  COFLAM[7] = 1.85596636E-02;
  COFLAM[8] = 8.22510925E+00;
  COFLAM[9] = -1.73468254E+00;
  COFLAM[10] = 4.32494580E-01;
  COFLAM[11] = -2.30931264E-02;
  COFLAM[12] = 1.07637397E+01;
  COFLAM[13] = -3.22891661E+00;
  COFLAM[14] = 7.02587028E-01;
  COFLAM[15] = -3.78546262E-02;
  COFLAM[16] = 1.76957806E+01;
  COFLAM[17] = -6.71274224E+00;
  COFLAM[18] = 1.26299563E+00;
  COFLAM[19] = -6.62747841E-02;
  COFLAM[20] = 4.91540446E+00;
  COFLAM[21] = -7.25650641E-01;
  COFLAM[22] = 3.28107625E-01;
  COFLAM[23] = -1.99735466E-02;
  COFLAM[24] = -1.08133063E+01;
  COFLAM[25] = 5.87459761E+00;
  COFLAM[26] = -5.86230968E-01;
  COFLAM[27] = 2.24413660E-02;
  COFLAM[28] = -4.10451012E+00;
  COFLAM[29] = 2.44778894E+00;
  COFLAM[30] = -2.59480363E-02;
  COFLAM[31] = -6.86823503E-03;
  COFLAM[32] = -4.03620887E+00;
  COFLAM[33] = 1.90761262E+00;
  COFLAM[34] = 1.17538704E-01;
  COFLAM[35] = -1.61049430E-02;
  COFLAM[36] = -1.28445991E+00;
  COFLAM[37] = 8.20566771E-01;
  COFLAM[38] = 2.59388942E-01;
  COFLAM[39] = -2.22604117E-02;
  COFLAM[40] = -2.11843099E+00;
  COFLAM[41] = 9.97190244E-01;
  COFLAM[42] = 2.61096662E-01;
  COFLAM[43] = -2.33522398E-02;
  COFLAM[44] = 1.42383887E+00;
  COFLAM[45] = 1.96214031E+00;
  COFLAM[46] = -1.79397210E-01;
  COFLAM[47] = 8.12672879E-03;
}

// Poly fits for the diffusion coefficients, dim NO*KK*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFD(amrex::Real* COFD)
{
  COFD[0] = -1.04285243E+01;
  COFD[1] = 2.23477534E+00;
  COFD[2] = -8.11809423E-02;
  COFD[3] = 3.77342041E-03;
  COFD[4] = -1.19370534E+01;
  COFD[5] = 2.99054752E+00;
  COFD[6] = -1.79624448E-01;
  COFD[7] = 8.03970815E-03;
  COFD[8] = -1.51395562E+01;
  COFD[9] = 4.36621619E+00;
  COFD[10] = -3.53866950E-01;
  COFD[11] = 1.54097445E-02;
  COFD[12] = -1.30060181E+01;
  COFD[13] = 2.98419920E+00;
  COFD[14] = -1.78783571E-01;
  COFD[15] = 8.00253963E-03;
  COFD[16] = -1.61362315E+01;
  COFD[17] = 4.35660780E+00;
  COFD[18] = -3.52659055E-01;
  COFD[19] = 1.53589488E-02;
  COFD[20] = -1.76381591E+01;
  COFD[21] = 4.34699543E+00;
  COFD[22] = -3.51450562E-01;
  COFD[23] = 1.53081221E-02;
  COFD[24] = -1.30102484E+01;
  COFD[25] = 2.98419920E+00;
  COFD[26] = -1.78783571E-01;
  COFD[27] = 8.00253963E-03;
  COFD[28] = -1.61384840E+01;
  COFD[29] = 4.35660780E+00;
  COFD[30] = -3.52659055E-01;
  COFD[31] = 1.53589488E-02;
  COFD[32] = -1.76552074E+01;
  COFD[33] = 4.34699543E+00;
  COFD[34] = -3.51450562E-01;
  COFD[35] = 1.53081221E-02;
  COFD[36] = -1.76728575E+01;
  COFD[37] = 4.34699543E+00;
  COFD[38] = -3.51450562E-01;
  COFD[39] = 1.53081221E-02;
  COFD[40] = -1.29544979E+01;
  COFD[41] = 2.96758239E+00;
  COFD[42] = -1.76586224E-01;
  COFD[43] = 7.90559536E-03;
  COFD[44] = -1.60530569E+01;
  COFD[45] = 4.33128165E+00;
  COFD[46] = -3.49478110E-01;
  COFD[47] = 1.52253415E-02;
  COFD[48] = -1.75898334E+01;
  COFD[49] = 4.32268865E+00;
  COFD[50] = -3.48416561E-01;
  COFD[51] = 1.51815419E-02;
  COFD[52] = -1.76080457E+01;
  COFD[53] = 4.32268865E+00;
  COFD[54] = -3.48416561E-01;
  COFD[55] = 1.51815419E-02;
  COFD[56] = -1.75618461E+01;
  COFD[57] = 4.30617914E+00;
  COFD[58] = -3.46490389E-01;
  COFD[59] = 1.51071405E-02;
  COFD[60] = -1.41038278E+01;
  COFD[61] = 3.35888991E+00;
  COFD[62] = -2.27939148E-01;
  COFD[63] = 1.01542945E-02;
  COFD[64] = -1.74057571E+01;
  COFD[65] = 4.76097419E+00;
  COFD[66] = -4.01175756E-01;
  COFD[67] = 1.73035102E-02;
  COFD[68] = -1.89981437E+01;
  COFD[69] = 4.75331782E+00;
  COFD[70] = -4.00274636E-01;
  COFD[71] = 1.72681883E-02;
  COFD[72] = -1.90201015E+01;
  COFD[73] = 4.75331782E+00;
  COFD[74] = -4.00274636E-01;
  COFD[75] = 1.72681883E-02;
  COFD[76] = -1.89668666E+01;
  COFD[77] = 4.73272421E+00;
  COFD[78] = -3.97843585E-01;
  COFD[79] = 1.71725994E-02;
  COFD[80] = -2.00360660E+01;
  COFD[81] = 4.98853197E+00;
  COFD[82] = -4.23032272E-01;
  COFD[83] = 1.79382596E-02;
  COFD[84] = -1.41052727E+01;
  COFD[85] = 3.35888991E+00;
  COFD[86] = -2.27939148E-01;
  COFD[87] = 1.01542945E-02;
  COFD[88] = -1.74065070E+01;
  COFD[89] = 4.76097419E+00;
  COFD[90] = -4.01175756E-01;
  COFD[91] = 1.73035102E-02;
  COFD[92] = -1.90051441E+01;
  COFD[93] = 4.75331782E+00;
  COFD[94] = -4.00274636E-01;
  COFD[95] = 1.72681883E-02;
  COFD[96] = -1.90274186E+01;
  COFD[97] = 4.75331782E+00;
  COFD[98] = -4.00274636E-01;
  COFD[99] = 1.72681883E-02;
  COFD[100] = -1.89744848E+01;
  COFD[101] = 4.73272421E+00;
  COFD[102] = -3.97843585E-01;
  COFD[103] = 1.71725994E-02;
  COFD[104] = -2.00458391E+01;
  COFD[105] = 4.98853197E+00;
  COFD[106] = -4.23032272E-01;
  COFD[107] = 1.79382596E-02;
  COFD[108] = -2.00558070E+01;
  COFD[109] = 4.98853197E+00;
  COFD[110] = -4.23032272E-01;
  COFD[111] = 1.79382596E-02;
  COFD[112] = -1.41066136E+01;
  COFD[113] = 3.35888991E+00;
  COFD[114] = -2.27939148E-01;
  COFD[115] = 1.01542945E-02;
  COFD[116] = -1.74072020E+01;
  COFD[117] = 4.76097419E+00;
  COFD[118] = -4.01175756E-01;
  COFD[119] = 1.73035102E-02;
  COFD[120] = -1.90117112E+01;
  COFD[121] = 4.75331782E+00;
  COFD[122] = -4.00274636E-01;
  COFD[123] = 1.72681883E-02;
  COFD[124] = -1.90342869E+01;
  COFD[125] = 4.75331782E+00;
  COFD[126] = -4.00274636E-01;
  COFD[127] = 1.72681883E-02;
  COFD[128] = -1.89816402E+01;
  COFD[129] = 4.73272421E+00;
  COFD[130] = -3.97843585E-01;
  COFD[131] = 1.71725994E-02;
  COFD[132] = -2.00550571E+01;
  COFD[133] = 4.98853197E+00;
  COFD[134] = -4.23032272E-01;
  COFD[135] = 1.79382596E-02;
  COFD[136] = -2.00652123E+01;
  COFD[137] = 4.98853197E+00;
  COFD[138] = -4.23032272E-01;
  COFD[139] = 1.79382596E-02;
  COFD[140] = -2.00747980E+01;
  COFD[141] = 4.98853197E+00;
  COFD[142] = -4.23032272E-01;
  COFD[143] = 1.79382596E-02;
  COFD[144] = -1.47832963E+01;
  COFD[145] = 3.61918603E+00;
  COFD[146] = -2.60701094E-01;
  COFD[147] = 1.15292870E-02;
  COFD[148] = -1.80153776E+01;
  COFD[149] = 4.95555480E+00;
  COFD[150] = -4.20418912E-01;
  COFD[151] = 1.78921168E-02;
  COFD[152] = -1.96435151E+01;
  COFD[153] = 4.95183665E+00;
  COFD[154] = -4.20101552E-01;
  COFD[155] = 1.78850963E-02;
  COFD[156] = -1.96663778E+01;
  COFD[157] = 4.95183665E+00;
  COFD[158] = -4.20101552E-01;
  COFD[159] = 1.78850963E-02;
  COFD[160] = -1.96315730E+01;
  COFD[161] = 4.93927762E+00;
  COFD[162] = -4.18850242E-01;
  COFD[163] = 1.78462242E-02;
  COFD[164] = -2.04588370E+01;
  COFD[165] = 5.07632177E+00;
  COFD[166] = -4.26053731E-01;
  COFD[167] = 1.77312885E-02;
  COFD[168] = -2.04691726E+01;
  COFD[169] = 5.07632177E+00;
  COFD[170] = -4.26053731E-01;
  COFD[171] = 1.77312885E-02;
  COFD[172] = -2.04789320E+01;
  COFD[173] = 5.07632177E+00;
  COFD[174] = -4.26053731E-01;
  COFD[175] = 1.77312885E-02;
  COFD[176] = -2.04987774E+01;
  COFD[177] = 4.99303150E+00;
  COFD[178] = -4.04298713E-01;
  COFD[179] = 1.63496647E-02;
  COFD[180] = -1.46102537E+01;
  COFD[181] = 3.52131229E+00;
  COFD[182] = -2.48399935E-01;
  COFD[183] = 1.10136492E-02;
  COFD[184] = -1.79184759E+01;
  COFD[185] = 4.89943438E+00;
  COFD[186] = -4.15602392E-01;
  COFD[187] = 1.77841166E-02;
  COFD[188] = -1.95301863E+01;
  COFD[189] = 4.89649396E+00;
  COFD[190] = -4.15388755E-01;
  COFD[191] = 1.77817581E-02;
  COFD[192] = -1.95533227E+01;
  COFD[193] = 4.89649396E+00;
  COFD[194] = -4.15388755E-01;
  COFD[195] = 1.77817581E-02;
  COFD[196] = -1.95331013E+01;
  COFD[197] = 4.88982264E+00;
  COFD[198] = -4.14974963E-01;
  COFD[199] = 1.77824366E-02;
  COFD[200] = -2.04074720E+01;
  COFD[201] = 5.05147258E+00;
  COFD[202] = -4.25999526E-01;
  COFD[203] = 1.78562806E-02;
  COFD[204] = -2.04179813E+01;
  COFD[205] = 5.05147258E+00;
  COFD[206] = -4.25999526E-01;
  COFD[207] = 1.78562806E-02;
  COFD[208] = -2.04279080E+01;
  COFD[209] = 5.05147258E+00;
  COFD[210] = -4.25999526E-01;
  COFD[211] = 1.78562806E-02;
  COFD[212] = -2.06376889E+01;
  COFD[213] = 5.05102272E+00;
  COFD[214] = -4.16130187E-01;
  COFD[215] = 1.70344431E-02;
  COFD[216] = -2.06778569E+01;
  COFD[217] = 5.06751728E+00;
  COFD[218] = -4.22067962E-01;
  COFD[219] = 1.74439059E-02;
  COFD[220] = -1.46113422E+01;
  COFD[221] = 3.52131229E+00;
  COFD[222] = -2.48399935E-01;
  COFD[223] = 1.10136492E-02;
  COFD[224] = -1.79190380E+01;
  COFD[225] = 4.89943438E+00;
  COFD[226] = -4.15602392E-01;
  COFD[227] = 1.77841166E-02;
  COFD[228] = -1.95356726E+01;
  COFD[229] = 4.89649396E+00;
  COFD[230] = -4.15388755E-01;
  COFD[231] = 1.77817581E-02;
  COFD[232] = -1.95590703E+01;
  COFD[233] = 4.89649396E+00;
  COFD[234] = -4.15388755E-01;
  COFD[235] = 1.77817581E-02;
  COFD[236] = -1.95390987E+01;
  COFD[237] = 4.88982264E+00;
  COFD[238] = -4.14974963E-01;
  COFD[239] = 1.77824366E-02;
  COFD[240] = -2.04152886E+01;
  COFD[241] = 5.05147258E+00;
  COFD[242] = -4.25999526E-01;
  COFD[243] = 1.78562806E-02;
  COFD[244] = -2.04259652E+01;
  COFD[245] = 5.05147258E+00;
  COFD[246] = -4.25999526E-01;
  COFD[247] = 1.78562806E-02;
  COFD[248] = -2.04360533E+01;
  COFD[249] = 5.05147258E+00;
  COFD[250] = -4.25999526E-01;
  COFD[251] = 1.78562806E-02;
  COFD[252] = -2.06459899E+01;
  COFD[253] = 5.05102272E+00;
  COFD[254] = -4.16130187E-01;
  COFD[255] = 1.70344431E-02;
  COFD[256] = -2.06863084E+01;
  COFD[257] = 5.06751728E+00;
  COFD[258] = -4.22067962E-01;
  COFD[259] = 1.74439059E-02;
  COFD[260] = -2.06949052E+01;
  COFD[261] = 5.06751728E+00;
  COFD[262] = -4.22067962E-01;
  COFD[263] = 1.74439059E-02;
  COFD[264] = -1.12314633E+01;
  COFD[265] = 2.35359960E+00;
  COFD[266] = -9.33663289E-02;
  COFD[267] = 4.13977197E-03;
  COFD[268] = -1.38406335E+01;
  COFD[269] = 3.58824515E+00;
  COFD[270] = -2.56761889E-01;
  COFD[271] = 1.13618125E-02;
  COFD[272] = -1.53581116E+01;
  COFD[273] = 3.58137147E+00;
  COFD[274] = -2.55890214E-01;
  COFD[275] = 1.13249037E-02;
  COFD[276] = -1.53738189E+01;
  COFD[277] = 3.58137147E+00;
  COFD[278] = -2.55890214E-01;
  COFD[279] = 1.13249037E-02;
  COFD[280] = -1.53238042E+01;
  COFD[281] = 3.56338026E+00;
  COFD[282] = -2.53615239E-01;
  COFD[283] = 1.12289204E-02;
  COFD[284] = -1.66790995E+01;
  COFD[285] = 3.99668524E+00;
  COFD[286] = -3.08443418E-01;
  COFD[287] = 1.35452935E-02;
  COFD[288] = -1.66854158E+01;
  COFD[289] = 3.99668524E+00;
  COFD[290] = -3.08443418E-01;
  COFD[291] = 1.35452935E-02;
  COFD[292] = -1.66913332E+01;
  COFD[293] = 3.99668524E+00;
  COFD[294] = -3.08443418E-01;
  COFD[295] = 1.35452935E-02;
  COFD[296] = -1.75322344E+01;
  COFD[297] = 4.30861106E+00;
  COFD[298] = -3.46772273E-01;
  COFD[299] = 1.51179387E-02;
  COFD[300] = -1.73763608E+01;
  COFD[301] = 4.22631945E+00;
  COFD[302] = -3.37571526E-01;
  COFD[303] = 1.47813966E-02;
  COFD[304] = -1.73812864E+01;
  COFD[305] = 4.22631945E+00;
  COFD[306] = -3.37571526E-01;
  COFD[307] = 1.47813966E-02;
  COFD[308] = -1.34113549E+01;
  COFD[309] = 2.91879827E+00;
  COFD[310] = -1.70127286E-01;
  COFD[311] = 7.62035353E-03;
}

// List of specs with small weight, dim NLITE
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKTDIF(int* KTDIF)
{
  KTDIF[0] = 0;
  KTDIF[1] = 1;
}

// Poly fits for thermal diff ratios, dim NO*NLITE*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFTD(amrex::Real* COFTD)
{
  COFTD[0] = 0.00000000E+00;
  COFTD[1] = 0.00000000E+00;
  COFTD[2] = 0.00000000E+00;
  COFTD[3] = 0.00000000E+00;
  COFTD[4] = -1.26715692E-01;
  COFTD[5] = -1.02530485E-04;
  COFTD[6] = 5.45604892E-08;
  COFTD[7] = -8.85181063E-12;
  COFTD[8] = 2.85592351E-01;
  COFTD[9] = 2.28508147E-04;
  COFTD[10] = -1.21625294E-07;
  COFTD[11] = 1.97426073E-11;
  COFTD[12] = 2.91259944E-01;
  COFTD[13] = 2.33042901E-04;
  COFTD[14] = -1.24038953E-07;
  COFTD[15] = 2.01344001E-11;
  COFTD[16] = 2.98971544E-01;
  COFTD[17] = 2.32229117E-04;
  COFTD[18] = -1.23674024E-07;
  COFTD[19] = 2.01028089E-11;
  COFTD[20] = 2.59746316E-01;
  COFTD[21] = 3.71757634E-04;
  COFTD[22] = -1.94052881E-07;
  COFTD[23] = 3.06513248E-11;
  COFTD[24] = 2.61381413E-01;
  COFTD[25] = 3.74097840E-04;
  COFTD[26] = -1.95274439E-07;
  COFTD[27] = 3.08442742E-11;
  COFTD[28] = 2.62903085E-01;
  COFTD[29] = 3.76275708E-04;
  COFTD[30] = -1.96411260E-07;
  COFTD[31] = 3.10238388E-11;
  COFTD[32] = 2.06620011E-01;
  COFTD[33] = 4.69836642E-04;
  COFTD[34] = -2.39995296E-07;
  COFTD[35] = 3.71484694E-11;
  COFTD[36] = 2.29104091E-01;
  COFTD[37] = 4.39065607E-04;
  COFTD[38] = -2.26132661E-07;
  COFTD[39] = 3.52557262E-11;
  COFTD[40] = 2.30177080E-01;
  COFTD[41] = 4.41121932E-04;
  COFTD[42] = -2.27191734E-07;
  COFTD[43] = 3.54208433E-11;
  COFTD[44] = 3.49586125E-01;
  COFTD[45] = 6.73633880E-05;
  COFTD[46] = -3.23863358E-08;
  COFTD[47] = 5.44517084E-12;
  COFTD[48] = 1.26715692E-01;
  COFTD[49] = 1.02530485E-04;
  COFTD[50] = -5.45604892E-08;
  COFTD[51] = 8.85181063E-12;
  COFTD[52] = 0.00000000E+00;
  COFTD[53] = 0.00000000E+00;
  COFTD[54] = 0.00000000E+00;
  COFTD[55] = 0.00000000E+00;
  COFTD[56] = 6.80927441E-02;
  COFTD[57] = 6.56712432E-04;
  COFTD[58] = -3.16405035E-07;
  COFTD[59] = 4.69044403E-11;
  COFTD[60] = 6.87552483E-02;
  COFTD[61] = 6.63101876E-04;
  COFTD[62] = -3.19483479E-07;
  COFTD[63] = 4.73607943E-11;
  COFTD[64] = 7.31513425E-02;
  COFTD[65] = 6.64292890E-04;
  COFTD[66] = -3.20609980E-07;
  COFTD[67] = 4.75829654E-11;
  COFTD[68] = -6.45981096E-03;
  COFTD[69] = 7.81160866E-04;
  COFTD[70] = -3.62599488E-07;
  COFTD[71] = 5.24664587E-11;
  COFTD[72] = -6.48001595E-03;
  COFTD[73] = 7.83604181E-04;
  COFTD[74] = -3.63733627E-07;
  COFTD[75] = 5.26305632E-11;
  COFTD[76] = -6.49876899E-03;
  COFTD[77] = 7.85871915E-04;
  COFTD[78] = -3.64786264E-07;
  COFTD[79] = 5.27828750E-11;
  COFTD[80] = -6.41008204E-02;
  COFTD[81] = 8.31129050E-04;
  COFTD[82] = -3.73286756E-07;
  COFTD[83] = 5.29104106E-11;
  COFTD[84] = -4.40902585E-02;
  COFTD[85] = 8.20054224E-04;
  COFTD[86] = -3.72898022E-07;
  COFTD[87] = 5.32635396E-11;
  COFTD[88] = -4.41930236E-02;
  COFTD[89] = 8.21965598E-04;
  COFTD[90] = -3.73767170E-07;
  COFTD[91] = 5.33876857E-11;
  COFTD[92] = 2.07582938E-01;
  COFTD[93] = 4.49532866E-04;
  COFTD[94] = -2.30167130E-07;
  COFTD[95] = 3.56991681E-11;
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

  // species 0: H2
  // Imported from NIST
  Tci[0] = 33.145000;
  ai[0] = 1e6 * 0.42748 * Rcst * Rcst * Tci[0] * Tci[0] /
          (2.015880 * 2.015880 * 12.964000);
  bi[0] = 0.08664 * Rcst * Tci[0] / (2.015880 * 12.964000);
  acentric_i[0] = -0.219000;

  // species 1: H
  Tci[1] = 1.316 * EPS[1];
  ai[1] = (5.55 * avogadro * avogadro * EPS[1] * boltzmann * 1e-24 * SIG[1] *
           SIG[1] * SIG[1]) /
          (wt[1] * wt[1]);
  bi[1] = 0.855 * avogadro * 1e-24 * SIG[1] * SIG[1] * SIG[1] / (wt[1]);
  acentric_i[1] = 0.0;

  // species 2: CH2
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

  // species 11: C(s)
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

  // species 0: H2
  // Imported from NIST
  sqrtOneOverTc[0] = 1.7369646834812e-01;
  sqrtAsti[0] = 2.4824229305911e+05;
  Bi[0] = 9.1362254169923e+00;
  Fomega[0] = 1.3798618890700e-01;

  // species 1: H
  // Computed from Lennard-Jones
  sqrtOneOverTc[1] = 7.2391579785311e-02;
  sqrtAsti[1] = 5.8451221185793e+05;
  Bi[1] = 4.4006634563410e+00;
  Fomega[1] = 4.8508000000000e-01;

  // species 2: CH2
  // Computed from Lennard-Jones
  sqrtOneOverTc[2] = 7.2642504557751e-02;
  sqrtAsti[2] = 1.0564067619908e+05;
  Bi[2] = 2.0142023516102e+00;
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

  // species 11: C(s)
  // Computed from Lennard-Jones
  sqrtOneOverTc[11] = 1.0316275924140e-01;
  sqrtAsti[11] = 7.0240119914876e+04;
  Bi[11] = 1.5377647194458e+00;
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
