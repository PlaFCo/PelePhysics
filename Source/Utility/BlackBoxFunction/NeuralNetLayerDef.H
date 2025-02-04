#ifndef NEURAL_NET_LD_H
#define NEURAL_NET_LD_H

#include <fstream>

namespace pele::physics {

class NNMath
{
  // GPU compatible math class
public:
  static constexpr amrex::Real inf = INFINITY;

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static amrex::Real
  clamp(const amrex::Real r, const amrex::Real min, const amrex::Real max)
  {
    amrex::Real r0 = (r < min) ? min : r;
    return (r0 > max) ? max : r0;
  }
};

class NNArrReaders
{
public:
  template <typename T1, typename T2>
  static void read_with_template(int size, T2* outdata, std::ifstream& file)
  {
    amrex::Vector<T1> fdata(size);
    file.read(reinterpret_cast<char*>(fdata.data()), size * sizeof(T1));
    for (int i = 0; i < size; i++) {
      outdata[i] = (T2)fdata[i];
    }
  }

  static void
  read_reals(int fsize, int arrsize, amrex::Real* outdata, std::ifstream& file)
  {
    if (fsize == 4) {
      read_with_template<float, amrex::Real>(arrsize, outdata, file);
    } else if (fsize == 8) {
      read_with_template<double, amrex::Real>(arrsize, outdata, file);
    } else {
      amrex::Error(
        "Cannot read data of size " + std::to_string(fsize) +
        " bytes into array of amrex::Reals.");
    }
  }

  static void
  read_ints(int isize, int arrsize, int* outdata, std::ifstream& file)
  {
    if (isize == 2) {
      read_with_template<short, int>(arrsize, outdata, file);
    }
    if (isize == 4) {
      read_with_template<int, int>(arrsize, outdata, file);
    } else {
      // Don't support longs
      amrex::Error(
        "Cannot read data of size " + std::to_string(isize) +
        " bytes into array of ints.");
    }
  }
};

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
int
dim_to_numel(const int* dim, size_t ndim)
{
  int numel = 1;
  for (size_t i = 0; i < ndim; i++) {
    numel *= dim[i];
  }
  return numel;
}

class NNLayer
{
public:
  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  // NOLINTNEXTLINE(modernize-use-equals-default)
  virtual ~NNLayer() {}

  // Template for CPU-only constructor to be used when reading in a file
  // NNLayer(std::vector<char>& types, std::vector<std::vector<int>>& sizes,
  // std::ifstream& file);

  // Inputs are size nin, outputs nout (see below). Outputs should be assumed to
  // be uninitialized
  AMREX_GPU_HOST_DEVICE
  virtual void operator()(
    const amrex::Real* inputs, amrex::Real* outputs, const int nin) = 0;
  // Return the number of floating point parameters for the layer
  AMREX_GPU_HOST_DEVICE
  virtual int nfparams() = 0;
  // Return the number of integer parameters for the layer, including sizes
  AMREX_GPU_HOST_DEVICE
  virtual int niparams() = 0;
  // Pack the layer into sections of Real and int arrays; assume memory already
  // allocated
  virtual void pack(amrex::Real* nnrd, int* nnid) = 0;

  // Dimension introduced by the layer, replacing the n2 from the input
  // dimensions. -1 implies no change in shape
  int nout = -1;
};

class LinearNNLayer : public NNLayer
{
public:
  LinearNNLayer(
    std::vector<char>& types,
    std::vector<std::vector<int>>& sizes,
    std::ifstream& file,
    int fsize,
    int isize)
  {
    AMREX_ALWAYS_ASSERT(types.size() == 2);
    AMREX_ALWAYS_ASSERT(sizes.size() == 2);
    AMREX_ALWAYS_ASSERT(types[0] == 'f');
    AMREX_ALWAYS_ASSERT(types[1] == 'f');
    AMREX_ALWAYS_ASSERT(sizes[0].size() == 2);
    AMREX_ALWAYS_ASSERT(sizes[1].size() == 1);
    AMREX_ALWAYS_ASSERT(sizes[0][0] == sizes[1][0]);

    owns_data = true;

    int wnumel = dim_to_numel(sizes[0].data(), sizes[0].size());
    int bnumel = sizes[1][0];
    weight = new amrex::Real[wnumel];
    bias = new amrex::Real[bnumel];
    wsize = new int[2];
    bsize = new int;

    NNArrReaders::read_reals(fsize, wnumel, weight, file);
    NNArrReaders::read_reals(isize, bnumel, bias, file);
    wsize[0] = sizes[0][0];
    wsize[1] = sizes[0][1];
    bsize[0] = sizes[1][0];

    nout = bnumel;
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  LinearNNLayer(amrex::Real* nnrd, int* nnid)
  {
    owns_data = false;

    wsize = nnid;
    bsize = nnid;
    weight = nnrd;
    bias = nnrd + wsize[0] * wsize[1];

    nout = bsize[0];
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  ~LinearNNLayer() override
  {
    if (owns_data) {
      delete[] weight;
      delete[] bias;
      delete[] wsize;
      delete bsize;
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void operator()(
    const amrex::Real* inputs, amrex::Real* outputs, const int nin) override
  {
    for (int j = 0; j < nout; j++) {
      outputs[j] = 0.0;
    }

    for (int j = 0; j < nout; j++) {
      for (int i = 0; i < nin; i++) {
        // Torch stores the weights as nout x nin instead of nin x nout
        outputs[j] += inputs[i] * weight[nin * j + i];
      }
      outputs[j] += bias[j];
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  int nfparams() override { return dim_to_numel(wsize, 2) + bsize[0]; }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  int niparams() override { return 2; }

  void pack(amrex::Real* nnrd, int* nnid) override
  {
    for (int j = 0; j < wsize[0]; j++) {
      for (int i = 0; i < wsize[1]; i++) {
        nnrd[wsize[1] * j + i] = weight[wsize[1] * j + i];
      }
    }

    int wnumel = wsize[0] * wsize[1];
    for (int i = wnumel; i < nfparams(); i++) {
      nnrd[i] = bias[i - wnumel];
    }

    nnid[0] = wsize[0];
    nnid[1] = wsize[1];
  }

private:
  bool owns_data = true;
  amrex::Real* weight;
  amrex::Real* bias;
  int* wsize;
  int* bsize;
};

class LeakyReluNNLayer : public NNLayer
{
public:
  LeakyReluNNLayer(
    std::vector<char>& types,
    std::vector<std::vector<int>>& sizes,
    std::ifstream& file,
    int fsize,
    int /*isize*/)
  {
    AMREX_ALWAYS_ASSERT(types.size() == 1);
    AMREX_ALWAYS_ASSERT(sizes.size() == 1);
    AMREX_ALWAYS_ASSERT(types[0] == 'f');
    AMREX_ALWAYS_ASSERT(sizes[0].size() == 1);

    owns_data = true;

    neg_slope = new amrex::Real;
    NNArrReaders::read_reals(fsize, 1, neg_slope, file);
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  LeakyReluNNLayer(amrex::Real* nnrd, int* /*nnid*/)
  {
    owns_data = false;
    neg_slope = nnrd;
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  ~LeakyReluNNLayer() override
  {
    if (owns_data) {
      delete neg_slope;
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void operator()(
    const amrex::Real* inputs, amrex::Real* outputs, const int nin) override
  {
    for (int i = 0; i < nin; i++) {
      outputs[i] = NNMath::clamp(inputs[i], 0.0, NNMath::inf) +
                   *neg_slope * NNMath::clamp(inputs[i], -NNMath::inf, 0.0);
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  int nfparams() override { return 1; }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  int niparams() override { return 0; }

  void pack(amrex::Real* nnrd, int* /*nnid*/) override
  {
    nnrd[0] = neg_slope[0];
  }

private:
  bool owns_data = true;
  amrex::Real* neg_slope;
};

class BatchNorm1dNNLayer : public NNLayer
{
public:
  BatchNorm1dNNLayer(
    std::vector<char>& types,
    std::vector<std::vector<int>>& sizes,
    std::ifstream& file,
    int fsize,
    int isize)
  {
    AMREX_ALWAYS_ASSERT(types.size() == 2);
    AMREX_ALWAYS_ASSERT(sizes.size() == 2);
    AMREX_ALWAYS_ASSERT(types[0] == 'f');
    AMREX_ALWAYS_ASSERT(types[1] == 'f');
    AMREX_ALWAYS_ASSERT(sizes[0].size() == 1);
    AMREX_ALWAYS_ASSERT(sizes[1].size() == 1);
    AMREX_ALWAYS_ASSERT(sizes[0][0] == sizes[1][0]);

    owns_data = true;

    int wnumel = sizes[0][0];
    int bnumel = sizes[1][0];
    weight = new amrex::Real[wnumel];
    bias = new amrex::Real[bnumel];
    wsize = new int;
    bsize = new int;

    NNArrReaders::read_reals(fsize, wnumel, weight, file);
    NNArrReaders::read_reals(isize, bnumel, bias, file);
    wsize[0] = sizes[0][0];
    bsize[0] = sizes[1][0];
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  BatchNorm1dNNLayer(amrex::Real* nnrd, int* nnid)
  {
    owns_data = false;

    wsize = nnid;
    bsize = nnid;
    weight = nnrd;
    bias = nnrd + wsize[0];
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  ~BatchNorm1dNNLayer() override
  {
    if (owns_data) {
      delete[] weight;
      delete[] bias;
      delete wsize;
      delete bsize;
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void operator()(
    const amrex::Real* inputs, amrex::Real* outputs, const int nin) override
  {
    for (int i = 0; i < nin; i++) {
      outputs[i] = inputs[i] * weight[i] + bias[i];
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  int nfparams() override { return wsize[0] + bsize[0]; }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  int niparams() override { return 1; }

  void pack(amrex::Real* nnrd, int* nnid) override
  {
    for (int i = 0; i < wsize[0]; i++) {
      nnrd[i] = weight[i];
    }

    for (int i = wsize[0]; i < nfparams(); i++) {
      nnrd[i] = bias[i - wsize[0]];
    }

    nnid[0] = wsize[0];
    nnid[1] = bsize[0];
  }

private:
  bool owns_data = true;
  amrex::Real* weight;
  amrex::Real* bias;
  int* wsize;
  int* bsize;
};

} // namespace pele::physics

#endif
