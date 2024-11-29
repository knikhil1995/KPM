#ifndef HamiltonianClass_h
#define HamiltonianClass_h

#include <headers.h>

namespace kpm
{
  class HamiltonianClass
  {
    //
    // methods
    //
  public:
    /**
     * @brief Destructor.
     */
    virtual ~HamiltonianClass() {};

    virtual void HX(thrust::device_vector<double> &src, const double scalarHX,
                    const double scalarY, const double scalarX,
                    thrust::device_vector<double> &dst) = 0;
    virtual unsigned long int nSites() = 0;
    virtual unsigned long int nSites1D() = 0;
    virtual cublasHandle_t *cublasHandle() = 0;
  };
} // namespace kpm
#endif
