#include "mechanism.H"
const int rmap[NUM_REACTIONS] = {
  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
  18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};

// Returns 0-based map of reaction order
void
GET_RMAP(int* _rmap)
{
  for (int j = 0; j < NUM_REACTIONS; ++j) {
    _rmap[j] = rmap[j];
  }
}

// Returns a count of gas species in a gas reaction, and their indices
// and stoichiometric coefficients. (Eq 50)
void
CKINU(const int i, int& nspec, int ki[], int nu[])
{
  const int ns[NUM_GAS_REACTIONS] = {3, 4, 2, 3, 4, 3, 3, 4, 3, 4, 3, 3,
                                     3, 3, 3, 3, 4, 4, 2, 4, 4, 4, 4, 3,
                                     3, 3, 3, 4, 3, 4, 3, 3, 4, 4, 3, 4};
  const int kiv[NUM_GAS_REACTIONS * 4] = {
    8,  7,  2,  0,  8,  2,  7,  1,  7,  15, 0,  0,  15, 13, 1,  0,  15, 7,
    14, 8,  14, 13, 2,  0,  7,  13, 1,  0,  13, 7,  12, 8,  13, 7,  23, 0,
    13, 2,  12, 1,  12, 11, 2,  0,  12, 7,  22, 0,  23, 22, 2,  0,  22, 21,
    2,  0,  20, 11, 7,  0,  20, 18, 2,  0,  20, 2,  18, 1,  22, 2,  21, 1,
    12, 30, 0,  0,  12, 13, 30, 2,  11, 2,  10, 1,  11, 7,  10, 8,  30, 2,
    27, 1,  27, 26, 2,  0,  10, 2,  11, 0,  11, 12, 27, 0,  7,  14, 2,  0,
    11, 27, 33, 2,  13, 12, 2,  0,  11, 14, 10, 15, 14, 2,  15, 0,  13, 11,
    1,  0,  12, 2,  11, 1,  11, 7,  19, 2,  22, 19, 1,  0,  22, 7,  21, 8};
  const int nuv[NUM_GAS_REACTIONS * 4] = {
    -1, 1,  1,  0,  -1, -1, 1,  1,  -2, 1,  0,  0,  -1, 1,  1,  0,  -1, -1,
    1,  1,  -1, 1,  1,  0,  -2, 1,  1,  0,  -1, -1, 1,  1,  -1, -1, 1,  0,
    -1, -1, 1,  1,  -1, 1,  1,  0,  -1, -1, 1,  0,  -1, 1,  1,  0,  -1, 1,
    1,  0,  -1, 1,  1,  0,  -1, 1,  1,  0,  -1, -1, 1,  1,  -1, -1, 1,  1,
    -2, 1,  0,  0,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1,
    1,  1,  -1, 1,  1,  0,  -1, -1, 1,  0,  -1, -1, 1,  0,  -2, 1,  1,  0,
    -1, -1, 1,  1,  -1, 1,  1,  0,  -1, -1, 1,  1,  -1, -1, 1,  0,  -1, 1,
    1,  0,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,  0,  -1, -1, 1,  1};
  if (i < 1) {
    // Return max num species per reaction
    nspec = 4;
  } else {
    if (i > NUM_GAS_REACTIONS) {
      nspec = -1;
    } else {
      nspec = ns[i - 1];
      for (int j = 0; j < nspec; ++j) {
        ki[j] = kiv[(i - 1) * 4 + j] + 1;
        nu[j] = nuv[(i - 1) * 4 + j];
      }
    }
  }
}

// Returns the progress rates of each reactions
// Given P, T, and mole fractions
void
CKKFKR(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real q_f[],
  amrex::Real q_r[])
{
  amrex::Real c[34]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 34; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  progressRateFR(q_f, q_r, c, T);

  // convert to chemkin units
  for (int id = 0; id < 36; ++id) {
    q_f[id] *= 1.0e-6;
    q_r[id] *= 1.0e-6;
  }
}

// compute the progress rate for each reaction
// USES progressRate : todo switch to GPU
void
progressRateFR(
  amrex::Real* q_f, amrex::Real* q_r, amrex::Real* sc, amrex::Real T)
{
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);
  // compute the Gibbs free energy
  amrex::Real g_RT[34];
  gibbs(g_RT, T);

  amrex::Real sc_qss[1];
  comp_qfqr(q_f, q_r, sc, sc_qss, T, invT, logT);
}

// save atomic weights into array
void
atomicWeight(amrex::Real* awt)
{
  awt[0] = 39.950000; // Ar
  awt[1] = 1.008000;  // H
  awt[2] = 12.011000; // C
}

// get atomic weight for all elements
void
CKAWT(amrex::Real* awt)
{
  atomicWeight(awt);
}

// Returns the elemental composition
// of the speciesi (mdim is num of elements)
void
CKNCF(int* ncf)
{
  int kd = 3;
  // Zero ncf
  for (int id = 0; id < kd * 34; ++id) {
    ncf[id] = 0;
  }

  // Ar
  ncf[0 * kd + 0] = 1; // Ar

  // H2
  ncf[1 * kd + 1] = 2; // H

  // H
  ncf[2 * kd + 1] = 1; // H

  // C
  ncf[3 * kd + 2] = 1; // C

  // CH
  ncf[4 * kd + 2] = 1; // C
  ncf[4 * kd + 1] = 1; // H

  // CH2
  ncf[5 * kd + 2] = 1; // C
  ncf[5 * kd + 1] = 2; // H

  // CH2(S)
  ncf[6 * kd + 2] = 1; // C
  ncf[6 * kd + 1] = 2; // H

  // CH3
  ncf[7 * kd + 2] = 1; // C
  ncf[7 * kd + 1] = 3; // H

  // CH4
  ncf[8 * kd + 2] = 1; // C
  ncf[8 * kd + 1] = 4; // H

  // C2
  ncf[9 * kd + 2] = 2; // C

  // C2H
  ncf[10 * kd + 2] = 2; // C
  ncf[10 * kd + 1] = 1; // H

  // C2H2
  ncf[11 * kd + 2] = 2; // C
  ncf[11 * kd + 1] = 2; // H

  // C2H3
  ncf[12 * kd + 2] = 2; // C
  ncf[12 * kd + 1] = 3; // H

  // C2H4
  ncf[13 * kd + 2] = 2; // C
  ncf[13 * kd + 1] = 4; // H

  // C2H5
  ncf[14 * kd + 2] = 2; // C
  ncf[14 * kd + 1] = 5; // H

  // C2H6
  ncf[15 * kd + 2] = 2; // C
  ncf[15 * kd + 1] = 6; // H

  // C3H2
  ncf[16 * kd + 2] = 3; // C
  ncf[16 * kd + 1] = 2; // H

  // C3H3
  ncf[17 * kd + 2] = 3; // C
  ncf[17 * kd + 1] = 3; // H

  // C3H4A
  ncf[18 * kd + 2] = 3; // C
  ncf[18 * kd + 1] = 4; // H

  // C3H4P
  ncf[19 * kd + 2] = 3; // C
  ncf[19 * kd + 1] = 4; // H

  // C3H5
  ncf[20 * kd + 2] = 3; // C
  ncf[20 * kd + 1] = 5; // H

  // C3H5A
  ncf[21 * kd + 2] = 3; // C
  ncf[21 * kd + 1] = 5; // H

  // C3H6
  ncf[22 * kd + 2] = 3; // C
  ncf[22 * kd + 1] = 6; // H

  // C3H7(n)
  ncf[23 * kd + 2] = 3; // C
  ncf[23 * kd + 1] = 7; // H

  // C4H2
  ncf[24 * kd + 2] = 4; // C
  ncf[24 * kd + 1] = 2; // H

  // H2CCCCH
  ncf[25 * kd + 2] = 4; // C
  ncf[25 * kd + 1] = 3; // H

  // C4H4(13)
  ncf[26 * kd + 2] = 4; // C
  ncf[26 * kd + 1] = 4; // H

  // C4H5
  ncf[27 * kd + 2] = 4; // C
  ncf[27 * kd + 1] = 5; // H

  // C4H5(n)
  ncf[28 * kd + 2] = 4; // C
  ncf[28 * kd + 1] = 5; // H

  // C4H5(i)
  ncf[29 * kd + 2] = 4; // C
  ncf[29 * kd + 1] = 5; // H

  // C4H6(13)
  ncf[30 * kd + 2] = 4; // C
  ncf[30 * kd + 1] = 6; // H

  // C4H8T2
  ncf[31 * kd + 2] = 4; // C
  ncf[31 * kd + 1] = 8; // H

  // C6H5
  ncf[32 * kd + 2] = 6; // C
  ncf[32 * kd + 1] = 5; // H

  // C6H6
  ncf[33 * kd + 2] = 6; // C
  ncf[33 * kd + 1] = 6; // H
}

// Returns the vector of strings of element names
void
CKSYME_STR(amrex::Vector<std::string>& ename)
{
  ename.resize(3);
  ename[0] = "Ar";
  ename[1] = "H";
  ename[2] = "C";
}

// Returns the vector of strings of species names
void
CKSYMS_STR(amrex::Vector<std::string>& kname)
{
  kname.resize(34);
  kname[0] = "Ar";
  kname[1] = "H2";
  kname[2] = "H";
  kname[3] = "C";
  kname[4] = "CH";
  kname[5] = "CH2";
  kname[6] = "CH2(S)";
  kname[7] = "CH3";
  kname[8] = "CH4";
  kname[9] = "C2";
  kname[10] = "C2H";
  kname[11] = "C2H2";
  kname[12] = "C2H3";
  kname[13] = "C2H4";
  kname[14] = "C2H5";
  kname[15] = "C2H6";
  kname[16] = "C3H2";
  kname[17] = "C3H3";
  kname[18] = "C3H4A";
  kname[19] = "C3H4P";
  kname[20] = "C3H5";
  kname[21] = "C3H5A";
  kname[22] = "C3H6";
  kname[23] = "C3H7(n)";
  kname[24] = "C4H2";
  kname[25] = "H2CCCCH";
  kname[26] = "C4H4(13)";
  kname[27] = "C4H5";
  kname[28] = "C4H5(n)";
  kname[29] = "C4H5(i)";
  kname[30] = "C4H6(13)";
  kname[31] = "C4H8T2";
  kname[32] = "C6H5";
  kname[33] = "C6H6";
}

// compute the sparsity pattern of the chemistry Jacobian
void
SPARSITY_INFO(int* nJdata, const int* consP, int NCELLS)
{
  amrex::GpuArray<amrex::Real, 1225> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 34> conc = {0.0};
  for (int n = 0; n < 34; n++) {
    conc[n] = 1.0 / 34.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  int nJdata_tmp = 0;
  for (int k = 0; k < 35; k++) {
    for (int l = 0; l < 35; l++) {
      if (Jac[35 * k + l] != 0.0) {
        nJdata_tmp = nJdata_tmp + 1;
      }
    }
  }

  *nJdata = NCELLS * nJdata_tmp;
}

// compute the sparsity pattern of the system Jacobian
void
SPARSITY_INFO_SYST(int* nJdata, const int* consP, int NCELLS)
{
  amrex::GpuArray<amrex::Real, 1225> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 34> conc = {0.0};
  for (int n = 0; n < 34; n++) {
    conc[n] = 1.0 / 34.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  int nJdata_tmp = 0;
  for (int k = 0; k < 35; k++) {
    for (int l = 0; l < 35; l++) {
      if (k == l) {
        nJdata_tmp = nJdata_tmp + 1;
      } else {
        if (Jac[35 * k + l] != 0.0) {
          nJdata_tmp = nJdata_tmp + 1;
        }
      }
    }
  }

  *nJdata = NCELLS * nJdata_tmp;
}

// compute the sparsity pattern of the simplified (for preconditioning) system
// Jacobian
void
SPARSITY_INFO_SYST_SIMPLIFIED(int* nJdata, const int* consP)
{
  amrex::GpuArray<amrex::Real, 1225> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 34> conc = {0.0};
  for (int n = 0; n < 34; n++) {
    conc[n] = 1.0 / 34.000000;
  }
  aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

  int nJdata_tmp = 0;
  for (int k = 0; k < 35; k++) {
    for (int l = 0; l < 35; l++) {
      if (k == l) {
        nJdata_tmp = nJdata_tmp + 1;
      } else {
        if (Jac[35 * k + l] != 0.0) {
          nJdata_tmp = nJdata_tmp + 1;
        }
      }
    }
  }

  nJdata[0] = nJdata_tmp;
}

// compute the sparsity pattern of the chemistry Jacobian in CSC format -- base
// 0
void
SPARSITY_PREPROC_CSC(int* rowVals, int* colPtrs, const int* consP, int NCELLS)
{
  amrex::GpuArray<amrex::Real, 1225> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 34> conc = {0.0};
  for (int n = 0; n < 34; n++) {
    conc[n] = 1.0 / 34.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  colPtrs[0] = 0;
  int nJdata_tmp = 0;
  for (int nc = 0; nc < NCELLS; nc++) {
    int offset_row = nc * 35;
    int offset_col = nc * 35;
    for (int k = 0; k < 35; k++) {
      for (int l = 0; l < 35; l++) {
        if (Jac[35 * k + l] != 0.0) {
          rowVals[nJdata_tmp] = l + offset_row;
          nJdata_tmp = nJdata_tmp + 1;
        }
      }
      colPtrs[offset_col + (k + 1)] = nJdata_tmp;
    }
  }
}

// compute the sparsity pattern of the chemistry Jacobian in CSR format -- base
// 0
void
SPARSITY_PREPROC_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base)
{
  amrex::GpuArray<amrex::Real, 1225> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 34> conc = {0.0};
  for (int n = 0; n < 34; n++) {
    conc[n] = 1.0 / 34.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  if (base == 1) {
    rowPtrs[0] = 1;
    int nJdata_tmp = 1;
    for (int nc = 0; nc < NCELLS; nc++) {
      int offset = nc * 35;
      for (int l = 0; l < 35; l++) {
        for (int k = 0; k < 35; k++) {
          if (Jac[35 * k + l] != 0.0) {
            colVals[nJdata_tmp - 1] = k + 1 + offset;
            nJdata_tmp = nJdata_tmp + 1;
          }
        }
        rowPtrs[offset + (l + 1)] = nJdata_tmp;
      }
    }
  } else {
    rowPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int nc = 0; nc < NCELLS; nc++) {
      int offset = nc * 35;
      for (int l = 0; l < 35; l++) {
        for (int k = 0; k < 35; k++) {
          if (Jac[35 * k + l] != 0.0) {
            colVals[nJdata_tmp] = k + offset;
            nJdata_tmp = nJdata_tmp + 1;
          }
        }
        rowPtrs[offset + (l + 1)] = nJdata_tmp;
      }
    }
  }
}

// compute the sparsity pattern of the system Jacobian
// CSR format BASE is user choice
void
SPARSITY_PREPROC_SYST_CSR(
  int* colVals, int* rowPtr, const int* consP, int NCELLS, int base)
{
  amrex::GpuArray<amrex::Real, 1225> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 34> conc = {0.0};
  for (int n = 0; n < 34; n++) {
    conc[n] = 1.0 / 34.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  if (base == 1) {
    rowPtr[0] = 1;
    int nJdata_tmp = 1;
    for (int nc = 0; nc < NCELLS; nc++) {
      int offset = nc * 35;
      for (int l = 0; l < 35; l++) {
        for (int k = 0; k < 35; k++) {
          if (k == l) {
            colVals[nJdata_tmp - 1] = l + 1 + offset;
            nJdata_tmp = nJdata_tmp + 1;
          } else {
            if (Jac[35 * k + l] != 0.0) {
              colVals[nJdata_tmp - 1] = k + 1 + offset;
              nJdata_tmp = nJdata_tmp + 1;
            }
          }
        }
        rowPtr[offset + (l + 1)] = nJdata_tmp;
      }
    }
  } else {
    rowPtr[0] = 0;
    int nJdata_tmp = 0;
    for (int nc = 0; nc < NCELLS; nc++) {
      int offset = nc * 35;
      for (int l = 0; l < 35; l++) {
        for (int k = 0; k < 35; k++) {
          if (k == l) {
            colVals[nJdata_tmp] = l + offset;
            nJdata_tmp = nJdata_tmp + 1;
          } else {
            if (Jac[35 * k + l] != 0.0) {
              colVals[nJdata_tmp] = k + offset;
              nJdata_tmp = nJdata_tmp + 1;
            }
          }
        }
        rowPtr[offset + (l + 1)] = nJdata_tmp;
      }
    }
  }
}

// compute the sparsity pattern of the simplified (for precond) system Jacobian
// on CPU BASE 0
void
SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(
  int* rowVals, int* colPtrs, int* indx, const int* consP)
{
  amrex::GpuArray<amrex::Real, 1225> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 34> conc = {0.0};
  for (int n = 0; n < 34; n++) {
    conc[n] = 1.0 / 34.000000;
  }
  aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

  colPtrs[0] = 0;
  int nJdata_tmp = 0;
  for (int k = 0; k < 35; k++) {
    for (int l = 0; l < 35; l++) {
      if (k == l) {
        rowVals[nJdata_tmp] = l;
        indx[nJdata_tmp] = 35 * k + l;
        nJdata_tmp = nJdata_tmp + 1;
      } else {
        if (Jac[35 * k + l] != 0.0) {
          rowVals[nJdata_tmp] = l;
          indx[nJdata_tmp] = 35 * k + l;
          nJdata_tmp = nJdata_tmp + 1;
        }
      }
    }
    colPtrs[k + 1] = nJdata_tmp;
  }
}

// compute the sparsity pattern of the simplified (for precond) system Jacobian
// CSR format BASE is under choice
void
SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(
  int* colVals, int* rowPtr, const int* consP, int base)
{
  amrex::GpuArray<amrex::Real, 1225> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 34> conc = {0.0};
  for (int n = 0; n < 34; n++) {
    conc[n] = 1.0 / 34.000000;
  }
  aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

  if (base == 1) {
    rowPtr[0] = 1;
    int nJdata_tmp = 1;
    for (int l = 0; l < 35; l++) {
      for (int k = 0; k < 35; k++) {
        if (k == l) {
          colVals[nJdata_tmp - 1] = l + 1;
          nJdata_tmp = nJdata_tmp + 1;
        } else {
          if (Jac[35 * k + l] != 0.0) {
            colVals[nJdata_tmp - 1] = k + 1;
            nJdata_tmp = nJdata_tmp + 1;
          }
        }
      }
      rowPtr[l + 1] = nJdata_tmp;
    }
  } else {
    rowPtr[0] = 0;
    int nJdata_tmp = 0;
    for (int l = 0; l < 35; l++) {
      for (int k = 0; k < 35; k++) {
        if (k == l) {
          colVals[nJdata_tmp] = l;
          nJdata_tmp = nJdata_tmp + 1;
        } else {
          if (Jac[35 * k + l] != 0.0) {
            colVals[nJdata_tmp] = k;
            nJdata_tmp = nJdata_tmp + 1;
          }
        }
      }
      rowPtr[l + 1] = nJdata_tmp;
    }
  }
}
