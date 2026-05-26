#include "mechanism.H"
const int rmap[NUM_REACTIONS] = {
  11, 12, 16, 26, 27, 28, 30, 31, 36, 38, 42, 43, 0,  1,  2,  3,
  4,  5,  6,  7,  8,  9,  10, 13, 14, 15, 17, 18, 19, 20, 21, 22,
  23, 24, 25, 29, 32, 33, 34, 35, 37, 39, 40, 41, 44, 45, 46, 47};

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
    4, 4, 4, 4, 3, 3, 4, 3, 4, 4, 4, 2, 2, 4, 4, 4, 3, 4, 3, 4, 4, 4, 3, 3,
    4, 4, 2, 2, 3, 3, 3, 3, 4, 3, 4, 4, 3, 4, 3, 4, 3, 4, 3, 3, 4, 4, 4, 4};
  const int kiv[NUM_GAS_REACTIONS * 4] = {
    5,  0,  6,  1, 2,  7, 3,  6, 7,  1,  6,  0, 3,  7,  12, 1, 4,  7,  6,  0,
    5,  7,  6,  0, 9,  7, 10, 6, 10, 1,  19, 0, 9,  0,  10, 1, 2,  4,  9,  1,
    2,  6,  10, 1, 1,  0, 0,  0, 1,  0,  0,  0, 3,  1,  2,  0, 4,  1,  3,  0,
    5,  1,  4,  1, 6,  1, 7,  0, 11, 1,  10, 0, 6,  13, 1,  0, 3,  4,  10, 1,
    3,  6,  11, 1, 11, 3, 10, 4, 4,  10, 0,  0, 4,  10, 1,  0, 4,  6,  12, 1,
    11, 4,  10, 6, 5,  4, 0,  0, 6,  14, 0,  0, 6,  4,  1,  0, 9,  11, 10, 0,
    19, 18, 1,  0, 10, 1, 11, 0, 10, 3,  9,  4, 10, 4,  16, 0, 10, 5,  15, 1,
    9,  10, 18, 1, 10, 9, 1,  0, 12, 1,  11, 0, 12, 1,  13, 0, 12, 3,  16, 1,
    12, 5,  17, 0, 12, 6, 11, 7, 12, 10, 0,  0, 12, 11, 1,  0, 14, 1,  13, 0,
    14, 3,  12, 6, 14, 5, 13, 6, 14, 6,  13, 7};
  const int nuv[NUM_GAS_REACTIONS * 4] = {
    -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 2, 0,
    -1, -1, 2, 0, -1, -1, 1, 1, -2, 1,  1, 0, -1, -1, 1, 1, -1, -1, 1, 1,
    -1, -1, 1, 1, -2, 1,  0, 0, -2, 1,  0, 0, -1, -1, 1, 1, -1, -1, 1, 1,
    -1, -1, 1, 1, -1, -1, 1, 0, -1, -1, 1, 1, -2, 1,  1, 0, -1, -1, 1, 1,
    -1, -1, 1, 1, -1, -1, 1, 1, -2, 1,  1, 0, -2, 1,  2, 0, -1, -1, 1, 1,
    -1, -1, 1, 1, -1, 1,  0, 0, -2, 1,  0, 0, -1, 1,  1, 0, -1, -1, 2, 0,
    -1, 1,  1, 0, -1, -1, 1, 0, -1, -1, 1, 1, -1, -1, 1, 0, -1, -1, 1, 1,
    -1, -1, 1, 1, -1, 1,  1, 0, -1, -1, 1, 1, -1, -1, 1, 0, -1, -1, 1, 1,
    -1, -1, 1, 0, -1, -1, 1, 1, -1, 1,  1, 0, -1, 1,  1, 0, -1, -1, 1, 1,
    -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1};
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
  amrex::Real c[20]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 20; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  progressRateFR(q_f, q_r, c, T);

  // convert to chemkin units
  for (int id = 0; id < 48; ++id) {
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
  amrex::Real g_RT[20];
  gibbs(g_RT, T);

  amrex::Real sc_qss[1];
  comp_qfqr(q_f, q_r, sc, sc_qss, T, invT, logT);
}

// Returns the vector of strings of element names
void
CKSYME_STR(amrex::Vector<std::string>& ename)
{
  ename.resize(2);
  ename[0] = "H";
  ename[1] = "C";
}

// Returns the vector of strings of species names
void
CKSYMS_STR(amrex::Vector<std::string>& kname)
{
  kname.resize(20);
  kname[0] = "H2";
  kname[1] = "H";
  kname[2] = "C";
  kname[3] = "CH";
  kname[4] = "CH2";
  kname[5] = "CH2(S)";
  kname[6] = "CH3";
  kname[7] = "CH4";
  kname[8] = "C2";
  kname[9] = "C2H";
  kname[10] = "C2H2";
  kname[11] = "C2H3";
  kname[12] = "C2H4";
  kname[13] = "C2H5";
  kname[14] = "C2H6";
  kname[15] = "C3H3";
  kname[16] = "C3H4";
  kname[17] = "C3H6";
  kname[18] = "C4H2";
  kname[19] = "H2CCCCH";
}

// compute the sparsity pattern of the chemistry Jacobian
void
SPARSITY_INFO(int* nJdata, const int* consP, int NCELLS)
{
  amrex::GpuArray<amrex::Real, 441> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 20> conc = {0.0};
  for (int n = 0; n < 20; n++) {
    conc[n] = 1.0 / 20.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  int nJdata_tmp = 0;
  for (int k = 0; k < 21; k++) {
    for (int l = 0; l < 21; l++) {
      if (Jac[21 * k + l] != 0.0) {
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
  amrex::GpuArray<amrex::Real, 441> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 20> conc = {0.0};
  for (int n = 0; n < 20; n++) {
    conc[n] = 1.0 / 20.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  int nJdata_tmp = 0;
  for (int k = 0; k < 21; k++) {
    for (int l = 0; l < 21; l++) {
      if (k == l) {
        nJdata_tmp = nJdata_tmp + 1;
      } else {
        if (Jac[21 * k + l] != 0.0) {
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
  amrex::GpuArray<amrex::Real, 441> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 20> conc = {0.0};
  for (int n = 0; n < 20; n++) {
    conc[n] = 1.0 / 20.000000;
  }
  aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

  int nJdata_tmp = 0;
  for (int k = 0; k < 21; k++) {
    for (int l = 0; l < 21; l++) {
      if (k == l) {
        nJdata_tmp = nJdata_tmp + 1;
      } else {
        if (Jac[21 * k + l] != 0.0) {
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
  amrex::GpuArray<amrex::Real, 441> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 20> conc = {0.0};
  for (int n = 0; n < 20; n++) {
    conc[n] = 1.0 / 20.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  colPtrs[0] = 0;
  int nJdata_tmp = 0;
  for (int nc = 0; nc < NCELLS; nc++) {
    int offset_row = nc * 21;
    int offset_col = nc * 21;
    for (int k = 0; k < 21; k++) {
      for (int l = 0; l < 21; l++) {
        if (Jac[21 * k + l] != 0.0) {
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
  amrex::GpuArray<amrex::Real, 441> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 20> conc = {0.0};
  for (int n = 0; n < 20; n++) {
    conc[n] = 1.0 / 20.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  if (base == 1) {
    rowPtrs[0] = 1;
    int nJdata_tmp = 1;
    for (int nc = 0; nc < NCELLS; nc++) {
      int offset = nc * 21;
      for (int l = 0; l < 21; l++) {
        for (int k = 0; k < 21; k++) {
          if (Jac[21 * k + l] != 0.0) {
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
      int offset = nc * 21;
      for (int l = 0; l < 21; l++) {
        for (int k = 0; k < 21; k++) {
          if (Jac[21 * k + l] != 0.0) {
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
  amrex::GpuArray<amrex::Real, 441> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 20> conc = {0.0};
  for (int n = 0; n < 20; n++) {
    conc[n] = 1.0 / 20.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  if (base == 1) {
    rowPtr[0] = 1;
    int nJdata_tmp = 1;
    for (int nc = 0; nc < NCELLS; nc++) {
      int offset = nc * 21;
      for (int l = 0; l < 21; l++) {
        for (int k = 0; k < 21; k++) {
          if (k == l) {
            colVals[nJdata_tmp - 1] = l + 1 + offset;
            nJdata_tmp = nJdata_tmp + 1;
          } else {
            if (Jac[21 * k + l] != 0.0) {
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
      int offset = nc * 21;
      for (int l = 0; l < 21; l++) {
        for (int k = 0; k < 21; k++) {
          if (k == l) {
            colVals[nJdata_tmp] = l + offset;
            nJdata_tmp = nJdata_tmp + 1;
          } else {
            if (Jac[21 * k + l] != 0.0) {
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
  amrex::GpuArray<amrex::Real, 441> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 20> conc = {0.0};
  for (int n = 0; n < 20; n++) {
    conc[n] = 1.0 / 20.000000;
  }
  aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

  colPtrs[0] = 0;
  int nJdata_tmp = 0;
  for (int k = 0; k < 21; k++) {
    for (int l = 0; l < 21; l++) {
      if (k == l) {
        rowVals[nJdata_tmp] = l;
        indx[nJdata_tmp] = 21 * k + l;
        nJdata_tmp = nJdata_tmp + 1;
      } else {
        if (Jac[21 * k + l] != 0.0) {
          rowVals[nJdata_tmp] = l;
          indx[nJdata_tmp] = 21 * k + l;
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
  amrex::GpuArray<amrex::Real, 441> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 20> conc = {0.0};
  for (int n = 0; n < 20; n++) {
    conc[n] = 1.0 / 20.000000;
  }
  aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

  if (base == 1) {
    rowPtr[0] = 1;
    int nJdata_tmp = 1;
    for (int l = 0; l < 21; l++) {
      for (int k = 0; k < 21; k++) {
        if (k == l) {
          colVals[nJdata_tmp - 1] = l + 1;
          nJdata_tmp = nJdata_tmp + 1;
        } else {
          if (Jac[21 * k + l] != 0.0) {
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
    for (int l = 0; l < 21; l++) {
      for (int k = 0; k < 21; k++) {
        if (k == l) {
          colVals[nJdata_tmp] = l;
          nJdata_tmp = nJdata_tmp + 1;
        } else {
          if (Jac[21 * k + l] != 0.0) {
            colVals[nJdata_tmp] = k;
            nJdata_tmp = nJdata_tmp + 1;
          }
        }
      }
      rowPtr[l + 1] = nJdata_tmp;
    }
  }
}
