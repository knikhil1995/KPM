#ifndef HamiltonianAndersonClass_H_
#define HamiltonianAndersonClass_H_

#include <Hamiltonian.h>
#include <cusparse.h>
namespace kpm
{
  class HamiltonianAnderson : public HamiltonianClass
  {
  public:
    HamiltonianAnderson(const unsigned int nDim, const unsigned long int nSites1D, const double disorder);

    ~HamiltonianAnderson();

    void HX(thrust::device_vector<double> &src, const double scalarHX,
            const double scalarY, const double scalarX,
            thrust::device_vector<double> &dst);
    unsigned long int nSites() { return d_nSites; }
    unsigned long int nSites1D() { return d_nSites1D; }
    cublasHandle_t *cublasHandle() { return &d_cublasHandle; }

  private:
    const unsigned int d_nDim;
    const unsigned long int d_nSites1D;
    const unsigned long int d_nSites;
    unsigned long int d_nVec;
    const double d_disorder;
    cusparseSpMatDescr_t d_sparseHamiltonianDescr;
    cusparseDnMatDescr_t d_srcDescr;
    cusparseDnMatDescr_t d_dstDescr;
    cusparseHandle_t d_cusparseHandle;
    cublasHandle_t d_cublasHandle;
    thrust::device_vector<double> d_cusparseBuffer;
    bool d_cusparsePreprocessDone;
    thrust::device_vector<unsigned long int> d_HamiltonianrowOffsets;
    thrust::device_vector<unsigned long int> d_HamiltoniancolIndices;
    thrust::device_vector<double> d_HamiltonianData;
  };
} // namespace kpm
#endif
