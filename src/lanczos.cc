#include <lanczos.h>

namespace kpm
{
  //
  // evaluate upper bound of the spectrum using k-step Lanczos iteration
  //
  std::pair<double, double>
  lanczosLowerUpperBoundEigenSpectrum(
      HamiltonianClass &HamiltonianOperator)
  {
    const unsigned int lanczosIterations = 80;
    double beta, betaNeg;

    double alpha, alphaNeg;

    //
    // generate random vector v
    //
    thrust::device_vector<double> X(HamiltonianOperator.nSites(), 0.0);
    thrust::device_vector<double> Y(HamiltonianOperator.nSites(), 0.0);
    thrust::device_vector<double> Z(HamiltonianOperator.nSites(), 0.0);
    thrust::host_vector<double> XHost(HamiltonianOperator.nSites(), 0.0);

    std::srand(0);
    for (unsigned long int i = 0; i < XHost.size(); i++)
      XHost[i] = ((double)std::rand()) / ((double)RAND_MAX);

    X = XHost;

    //
    // evaluate l2 norm
    //
    double XNorm;
    cublasDnrm2(*HamiltonianOperator.cublasHandle(),
                X.size(),
                thrust::raw_pointer_cast(X.data()),
                1,
                &XNorm);
    XNorm = 1.0 / XNorm;
    cublasDscal(*HamiltonianOperator.cublasHandle(),
                X.size(),
                &XNorm,
                thrust::raw_pointer_cast(X.data()),
                1);

    //
    // call matrix times X
    //
    HamiltonianOperator.HX(X, 1.0, 0.0, 0.0, Y);

    cublasDdot(*HamiltonianOperator.cublasHandle(),
               X.size(),
               thrust::raw_pointer_cast(X.data()),
               1,
               thrust::raw_pointer_cast(Y.data()),
               1,
               &alpha);

    alphaNeg = -alpha;
    cublasDaxpy(*HamiltonianOperator.cublasHandle(),
                X.size(),
                &alphaNeg,
                thrust::raw_pointer_cast(X.data()),
                1,
                thrust::raw_pointer_cast(Y.data()),
                1);

    std::vector<double> Tlanczos(lanczosIterations * lanczosIterations, 0.0);

    Tlanczos[0] = alpha;
    unsigned index = 0;

    // filling only lower triangular part
    for (unsigned int j = 1; j < lanczosIterations; j++)
    {
      cublasDnrm2(*HamiltonianOperator.cublasHandle(),
                  Y.size(),
                  thrust::raw_pointer_cast(Y.data()),
                  1,
                  &beta);
      Z = X;
      thrust::transform(Y.begin(), Y.end(), X.begin(),
                        [beta] __device__(double &x)
                        { return x / beta; });

      HamiltonianOperator.HX(X, 1.0, 0.0, 0.0, Y);
      alphaNeg = -beta;
      cublasDaxpy(*HamiltonianOperator.cublasHandle(),
                  X.size(),
                  &alphaNeg,
                  thrust::raw_pointer_cast(Z.data()),
                  1,
                  thrust::raw_pointer_cast(Y.data()),
                  1);

      cublasDdot(*HamiltonianOperator.cublasHandle(),
                 X.size(),
                 thrust::raw_pointer_cast(X.data()),
                 1,
                 thrust::raw_pointer_cast(Y.data()),
                 1,
                 &alpha);
      alphaNeg = -alpha;
      cublasDaxpy(*HamiltonianOperator.cublasHandle(),
                  X.size(),
                  &alphaNeg,
                  thrust::raw_pointer_cast(X.data()),
                  1,
                  thrust::raw_pointer_cast(Y.data()),
                  1);

      index += 1;
      Tlanczos[index] = beta;
      index += lanczosIterations;
      Tlanczos[index] = alpha;
    }

    // eigen decomposition to find max eigen value of T matrix
    std::vector<double> eigenValuesT(lanczosIterations);
    char jobz = 'N', uplo = 'L';
    const unsigned int n = lanczosIterations, lda = lanczosIterations;
    int info;
    const unsigned int lwork = 1 + 6 * n + 2 * n * n, liwork = 3 + 5 * n;
    std::vector<int> iwork(liwork, 0);

    std::vector<double> work(lwork, 0.0);
    dsyevd_(&jobz,
            &uplo,
            &n,
            &Tlanczos[0],
            &lda,
            &eigenValuesT[0],
            &work[0],
            &lwork,
            &iwork[0],
            &liwork,
            &info);

    std::sort(eigenValuesT.begin(), eigenValuesT.end());
    //
    double YNorm;
    cublasDnrm2(*HamiltonianOperator.cublasHandle(),
                Y.size(),
                thrust::raw_pointer_cast(Y.data()),
                1,
                &YNorm);
    double lowerBound = std::floor(eigenValuesT[0] - YNorm);
    double upperBound =
        std::ceil(eigenValuesT[lanczosIterations - 1] + YNorm);

    if (true)
    {
      std::cout << "bUp1: " << eigenValuesT[lanczosIterations - 1]
                << ", fvector norm: " << YNorm << std::endl;
      std::cout << "aLow: " << eigenValuesT[0] << std::endl;
      std::cout << "boundL: " << lowerBound << std::endl;
      std::cout << "boundU: " << upperBound << std::endl;
    }

    return (std::make_pair(lowerBound, upperBound));
  }
}