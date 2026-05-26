#include "mechanism.H"
const int rmap[NUM_REACTIONS] = {
  0,  8,  9,  27, 28, 44, 46, 72, 78, 80, 81, 82, 84, 1,  2,  3,  4,  5,
  6,  7,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
  26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 47,
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
  66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 79, 83, 85, 86, 87};

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
  const int ns[NUM_GAS_REACTIONS] = {
    2, 4, 4, 3, 3, 2, 3, 3, 3, 3, 4, 3, 3, 4, 4, 4, 3, 4, 3, 4, 4, 3,
    4, 4, 3, 4, 4, 3, 3, 3, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    3, 4, 3, 4, 3, 4, 2, 4, 4, 4, 4, 4, 3, 3, 3, 4, 3, 4, 3, 3, 3, 3,
    4, 2, 4, 2, 3, 3, 3, 3, 4, 4, 2, 3, 3, 4, 3, 3, 2, 3, 3, 4, 3, 5};
  const int kiv[NUM_GAS_REACTIONS * 5] = {
    2,  1,  0,  0,  0, 10, 2,  9,  1,  0, 6,  10, 15, 2,  0, 7,  10, 9,  0,  0,
    9,  2,  10, 0,  0, 9,  17, 0,  0,  0, 9,  16, 2,  0,  0, 9,  15, 1,  0,  0,
    9,  7,  2,  0,  0, 9,  6,  1,  0,  0, 7,  2,  6,  1,  0, 7,  13, 1,  0,  0,
    7,  13, 2,  0,  0, 7,  9,  15, 2,  0, 12, 7,  13, 6,  0, 14, 7,  13, 9,  0,
    12, 2,  13, 0,  0, 12, 10, 13, 9,  0, 12, 14, 13, 0,  0, 12, 16, 13, 15, 0,
    13, 2,  12, 1,  0, 14, 13, 2,  0,  0, 14, 2,  13, 1,  0, 14, 9,  13, 10, 0,
    14, 13, 15, 0,  0, 15, 9,  14, 10, 0, 15, 2,  14, 1,  0, 15, 13, 1,  0,  0,
    15, 14, 2,  0,  0, 16, 15, 2,  0,  0, 13, 16, 12, 17, 0, 16, 9,  15, 10, 0,
    16, 15, 17, 0,  0, 17, 2,  16, 1,  0, 17, 7,  16, 9,  0, 17, 9,  16, 10, 0,
    13, 6,  18, 2,  0, 13, 7,  19, 2,  0, 12, 9,  19, 2,  0, 20, 2,  19, 1,  0,
    12, 20, 13, 19, 0, 21, 2,  19, 1,  0, 12, 21, 13, 19, 0, 28, 2,  19, 9,  0,
    19, 2,  20, 0,  0, 13, 9,  20, 2,  0, 19, 2,  21, 0,  0, 13, 9,  21, 2,  0,
    13, 9,  22, 0,  0, 13, 14, 26, 2,  0, 13, 26, 0,  0,  0, 12, 15, 26, 2,  0,
    27, 2,  26, 1,  0, 19, 7,  26, 2,  0, 18, 9,  26, 2,  0, 28, 2,  26, 1,  0,
    26, 2,  27, 0,  0, 13, 14, 27, 0,  0, 14, 27, 2,  0,  0, 29, 2,  27, 1,  0,
    29, 27, 2,  0,  0, 29, 2,  28, 1,  0, 29, 28, 2,  0,  0, 26, 2,  28, 0,  0,
    13, 14, 28, 0,  0, 14, 28, 2,  0,  0, 27, 2,  28, 2,  0, 27, 28, 0,  0,  0,
    14, 15, 29, 2,  0, 14, 29, 0,  0,  0, 30, 22, 9,  0,  0, 30, 29, 1,  0,  0,
    31, 2,  32, 0,  0, 13, 26, 32, 0,  0, 13, 27, 32, 2,  0, 14, 27, 32, 1,  0,
    19, 32, 0,  0,  0, 7,  6,  9,  0,  0, 7,  6,  2,  0,  0, 3,  1,  6,  2,  0,
    6,  3,  2,  0,  0, 7,  3,  1,  0,  0, 11, 3,  0,  0,  0, 12, 11, 13, 0,  0,
    12, 11, 2,  0,  0, 12, 2,  11, 1,  0, 32, 5,  1,  0,  0, 13, 5,  4,  5,  1};
  const int nuv[NUM_GAS_REACTIONS * 5] = {
    -2, 1,  0, 0, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0, -1, -1, 2, 0, 0,
    -1, -1, 1, 0, 0, -2, 1,  0, 0, 0, -2, 1,  1, 0, 0, -2, 1,  1, 0, 0,
    -1, 1,  1, 0, 0, -1, 1,  1, 0, 0, -1, -1, 1, 1, 0, -2, 1,  1, 0, 0,
    -2, 1,  2, 0, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0,
    -1, -1, 1, 0, 0, -1, -1, 1, 1, 0, -1, -1, 2, 0, 0, -1, -1, 1, 1, 0,
    -1, -1, 1, 1, 0, -1, 1,  1, 0, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0,
    -2, 1,  1, 0, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0, -1, 1,  1, 0, 0,
    -1, 1,  1, 0, 0, -1, 1,  1, 0, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0,
    -2, 1,  1, 0, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0,
    -1, -1, 1, 1, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0,
    -1, -1, 1, 1, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0,
    -1, -1, 1, 0, 0, -1, -1, 1, 1, 0, -1, -1, 1, 0, 0, -1, -1, 1, 1, 0,
    -1, -1, 1, 0, 0, -1, -1, 1, 1, 0, -2, 1,  0, 0, 0, -1, -1, 1, 1, 0,
    -1, -1, 1, 1, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0,
    -1, -1, 1, 0, 0, -1, -1, 1, 0, 0, -2, 1,  1, 0, 0, -1, -1, 1, 1, 0,
    -1, 1,  1, 0, 0, -1, -1, 1, 1, 0, -1, 1,  1, 0, 0, -1, -1, 1, 0, 0,
    -1, -1, 1, 0, 0, -2, 1,  1, 0, 0, -1, -1, 1, 1, 0, -1, 1,  0, 0, 0,
    -1, -1, 1, 1, 0, -2, 1,  0, 0, 0, -1, 1,  1, 0, 0, -1, 1,  1, 0, 0,
    -1, -1, 1, 0, 0, -1, -1, 1, 0, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0,
    -2, 1,  0, 0, 0, -2, 1,  1, 0, 0, -1, 1,  1, 0, 0, -1, -1, 1, 1, 0,
    -1, 1,  1, 0, 0, -1, 1,  1, 0, 0, -1, 2,  0, 0, 0, -2, 1,  1, 0, 0,
    -1, 1,  1, 0, 0, -1, -1, 1, 1, 0, -1, 6,  3, 0, 0, -1, -1, 2, 1, 1};
  if (i < 1) {
    // Return max num species per reaction
    nspec = 5;
  } else {
    if (i > NUM_GAS_REACTIONS) {
      nspec = -1;
    } else {
      nspec = ns[i - 1];
      for (int j = 0; j < nspec; ++j) {
        ki[j] = kiv[(i - 1) * 5 + j] + 1;
        nu[j] = nuv[(i - 1) * 5 + j];
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
  amrex::Real c[33]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 33; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  progressRateFR(q_f, q_r, c, T);

  // convert to chemkin units
  for (int id = 0; id < 88; ++id) {
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
  amrex::Real g_RT[33];
  gibbs(g_RT, T);

  amrex::Real sc_qss[1];
  comp_qfqr(q_f, q_r, sc, sc_qss, T, invT, logT);
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
  kname.resize(33);
  kname[0] = "Ar";
  kname[1] = "H2";
  kname[2] = "H";
  kname[3] = "C";
  kname[4] = "C(s)";
  kname[5] = "Csoot";
  kname[6] = "CH";
  kname[7] = "CH2";
  kname[8] = "CH2(S)";
  kname[9] = "CH3";
  kname[10] = "CH4";
  kname[11] = "C2";
  kname[12] = "C2H";
  kname[13] = "C2H2";
  kname[14] = "C2H3";
  kname[15] = "C2H4";
  kname[16] = "C2H5";
  kname[17] = "C2H6";
  kname[18] = "C3H2";
  kname[19] = "C3H3";
  kname[20] = "C3H4A";
  kname[21] = "C3H4P";
  kname[22] = "C3H5A";
  kname[23] = "C3H6";
  kname[24] = "C4H2";
  kname[25] = "H2CCCCH";
  kname[26] = "C4H4(13)";
  kname[27] = "C4H5(n)";
  kname[28] = "C4H5(i)";
  kname[29] = "C4H6(13)";
  kname[30] = "C4H8T2";
  kname[31] = "C6H5";
  kname[32] = "C6H6";
}

// compute the sparsity pattern of the chemistry Jacobian
void
SPARSITY_INFO(int* nJdata, const int* consP, int NCELLS)
{
  amrex::GpuArray<amrex::Real, 1156> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 33> conc = {0.0};
  for (int n = 0; n < 33; n++) {
    conc[n] = 1.0 / 33.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  int nJdata_tmp = 0;
  for (int k = 0; k < 34; k++) {
    for (int l = 0; l < 34; l++) {
      if (Jac[34 * k + l] != 0.0) {
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
  amrex::GpuArray<amrex::Real, 1156> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 33> conc = {0.0};
  for (int n = 0; n < 33; n++) {
    conc[n] = 1.0 / 33.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  int nJdata_tmp = 0;
  for (int k = 0; k < 34; k++) {
    for (int l = 0; l < 34; l++) {
      if (k == l) {
        nJdata_tmp = nJdata_tmp + 1;
      } else {
        if (Jac[34 * k + l] != 0.0) {
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
  amrex::GpuArray<amrex::Real, 1156> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 33> conc = {0.0};
  for (int n = 0; n < 33; n++) {
    conc[n] = 1.0 / 33.000000;
  }
  aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

  int nJdata_tmp = 0;
  for (int k = 0; k < 34; k++) {
    for (int l = 0; l < 34; l++) {
      if (k == l) {
        nJdata_tmp = nJdata_tmp + 1;
      } else {
        if (Jac[34 * k + l] != 0.0) {
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
  amrex::GpuArray<amrex::Real, 1156> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 33> conc = {0.0};
  for (int n = 0; n < 33; n++) {
    conc[n] = 1.0 / 33.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  colPtrs[0] = 0;
  int nJdata_tmp = 0;
  for (int nc = 0; nc < NCELLS; nc++) {
    int offset_row = nc * 34;
    int offset_col = nc * 34;
    for (int k = 0; k < 34; k++) {
      for (int l = 0; l < 34; l++) {
        if (Jac[34 * k + l] != 0.0) {
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
  amrex::GpuArray<amrex::Real, 1156> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 33> conc = {0.0};
  for (int n = 0; n < 33; n++) {
    conc[n] = 1.0 / 33.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  if (base == 1) {
    rowPtrs[0] = 1;
    int nJdata_tmp = 1;
    for (int nc = 0; nc < NCELLS; nc++) {
      int offset = nc * 34;
      for (int l = 0; l < 34; l++) {
        for (int k = 0; k < 34; k++) {
          if (Jac[34 * k + l] != 0.0) {
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
      int offset = nc * 34;
      for (int l = 0; l < 34; l++) {
        for (int k = 0; k < 34; k++) {
          if (Jac[34 * k + l] != 0.0) {
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
  amrex::GpuArray<amrex::Real, 1156> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 33> conc = {0.0};
  for (int n = 0; n < 33; n++) {
    conc[n] = 1.0 / 33.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  if (base == 1) {
    rowPtr[0] = 1;
    int nJdata_tmp = 1;
    for (int nc = 0; nc < NCELLS; nc++) {
      int offset = nc * 34;
      for (int l = 0; l < 34; l++) {
        for (int k = 0; k < 34; k++) {
          if (k == l) {
            colVals[nJdata_tmp - 1] = l + 1 + offset;
            nJdata_tmp = nJdata_tmp + 1;
          } else {
            if (Jac[34 * k + l] != 0.0) {
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
      int offset = nc * 34;
      for (int l = 0; l < 34; l++) {
        for (int k = 0; k < 34; k++) {
          if (k == l) {
            colVals[nJdata_tmp] = l + offset;
            nJdata_tmp = nJdata_tmp + 1;
          } else {
            if (Jac[34 * k + l] != 0.0) {
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
  amrex::GpuArray<amrex::Real, 1156> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 33> conc = {0.0};
  for (int n = 0; n < 33; n++) {
    conc[n] = 1.0 / 33.000000;
  }
  aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

  colPtrs[0] = 0;
  int nJdata_tmp = 0;
  for (int k = 0; k < 34; k++) {
    for (int l = 0; l < 34; l++) {
      if (k == l) {
        rowVals[nJdata_tmp] = l;
        indx[nJdata_tmp] = 34 * k + l;
        nJdata_tmp = nJdata_tmp + 1;
      } else {
        if (Jac[34 * k + l] != 0.0) {
          rowVals[nJdata_tmp] = l;
          indx[nJdata_tmp] = 34 * k + l;
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
  amrex::GpuArray<amrex::Real, 1156> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 33> conc = {0.0};
  for (int n = 0; n < 33; n++) {
    conc[n] = 1.0 / 33.000000;
  }
  aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

  if (base == 1) {
    rowPtr[0] = 1;
    int nJdata_tmp = 1;
    for (int l = 0; l < 34; l++) {
      for (int k = 0; k < 34; k++) {
        if (k == l) {
          colVals[nJdata_tmp - 1] = l + 1;
          nJdata_tmp = nJdata_tmp + 1;
        } else {
          if (Jac[34 * k + l] != 0.0) {
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
    for (int l = 0; l < 34; l++) {
      for (int k = 0; k < 34; k++) {
        if (k == l) {
          colVals[nJdata_tmp] = l;
          nJdata_tmp = nJdata_tmp + 1;
        } else {
          if (Jac[34 * k + l] != 0.0) {
            colVals[nJdata_tmp] = k;
            nJdata_tmp = nJdata_tmp + 1;
          }
        }
      }
      rowPtr[l + 1] = nJdata_tmp;
    }
  }
}
