#include <moments.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cublas_v2.h>

namespace kpm
{
  std::vector<double> computeTotalDensityOfStates(
      HamiltonianClass &HamiltonianOperator,
      const unsigned long int numMoments,
      const unsigned long int numPoints,
      const double a,
      const double b,
      const unsigned long int numSamples)
  {
    const unsigned long int nSites = HamiltonianOperator.nSites();
    const double pi = M_PI;
    const double c = (a + b) / 2.0;
    const double e = (b - a) / (2.0 - 1e-10);
    // Chebyshev moments
    std::vector<double> moments(numMoments, 0.0);

    // Initialize thrust device vectors for KPM
    thrust::device_vector<double> T0(nSites * numSamples, 0.0);
    thrust::device_vector<double> T1(nSites * numSamples, 0.0);
    thrust::device_vector<double> workspace(nSites * numSamples, 0.0);

    // Compute scaling and shifting parameters for mapping spectrum to [-1, 1]
    double scale = 1.0 / e;
    double shift = -c / e;

    // Generate random initial matrix (numSamples columns)
    thrust::host_vector<double> randomMatrix(nSites * numSamples);
    thrust::device_vector<double> randomMatrixDevice;
    thrust::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
    thrust::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (unsigned long int i = 0; i < nSites * numSamples; ++i)
    {
      randomMatrix[i] = dist(rng);
    }
    thrust::copy(randomMatrix.begin(), randomMatrix.end(), T0.begin());

    // Normalize T0 columns using cuBLAS
    thrust::device_vector<double> norms(numSamples, 0.0);
    thrust::device_vector<double> ones(nSites, 1.0);
    thrust::transform(T0.begin(), T0.end(), T0.begin(),
                      [] __device__(double x)
                      { return x * x; });
    double alpha = 1.0, beta = 0.0;
    cublasDgemv(*HamiltonianOperator.cublasHandle(),
                CUBLAS_OP_T,
                nSites,
                numSamples,
                &alpha,
                thrust::raw_pointer_cast(T0.data()),
                nSites,
                thrust::raw_pointer_cast(ones.data()),
                1,
                &beta,
                thrust::raw_pointer_cast(norms.data()),
                1);
    thrust::transform(norms.begin(), norms.end(), norms.begin(),
                      [] __device__(double x)
                      { return 1.0 / sqrt(x); });
    thrust::copy(randomMatrix.begin(), randomMatrix.end(), T0.begin());
    cublasDdgmm(*HamiltonianOperator.cublasHandle(),
                CUBLAS_SIDE_RIGHT,
                nSites,
                numSamples,
                thrust::raw_pointer_cast(T0.data()),
                nSites,
                thrust::raw_pointer_cast(norms.data()),
                1,
                thrust::raw_pointer_cast(T0.data()),
                nSites);
    randomMatrixDevice = T0;
    // Compute T_1 = scaled H * T_0
    HamiltonianOperator.HX(T0, scale, 0.0, shift, T1);
    moments[0] = 1.0;
    // Compute moments
    for (unsigned long int m = 1; m < numMoments; ++m)
    {
      thrust::transform(randomMatrixDevice.begin(),
                        randomMatrixDevice.end(),
                        T1.begin(),
                        workspace.begin(),
                        thrust::multiplies<double>());
      thrust::device_vector<double> sampleSums(numSamples, 0.0);
      cublasDgemv(*HamiltonianOperator.cublasHandle(),
                  CUBLAS_OP_T,
                  nSites,
                  numSamples,
                  &alpha,
                  thrust::raw_pointer_cast(workspace.data()),
                  nSites,
                  thrust::raw_pointer_cast(ones.data()),
                  1,
                  &beta,
                  thrust::raw_pointer_cast(sampleSums.data()),
                  1);
      moments[m] = thrust::reduce(sampleSums.begin(), sampleSums.end(), 0.0) / numSamples;

      if (m < numMoments - 1)
      {
        HamiltonianOperator.HX(T1, 2.0 * scale, -1.0, -2.0 * shift, T0);
        T0.swap(T1);
      }
    }

    // Apply Jackson kernel to moments
    std::vector<double> kernel(numMoments);
    for (unsigned long int m = 0; m < numMoments; ++m)
    {
      kernel[m] = (numMoments - m + 1) * std::cos(pi * m / (numMoments + 1.0)) +
                  std::sin(pi * m / (numMoments + 1.0)) / std::tan(pi / (numMoments + 1.0));
      kernel[m] /= numMoments + 1.0;
    }
    for (unsigned long int m = 0; m < numMoments; ++m)
    {
      moments[m] *= kernel[m];
      std::cout << "Moment " << m << ": " << moments[m] << " " << kernel[m] << std::endl;
    }

    // Use cuFFT to compute DCT-III
    thrust::host_vector<cufftDoubleComplex> fftInputHost(numPoints, make_cuDoubleComplex(0.0, 0.0));
    fftInputHost[0].x = moments[0];
    for (unsigned long int i = 1; i < numMoments; ++i)
    {
      fftInputHost[i].x = 2.0 * moments[i] * std::cos(pi * i / 2.0 / numPoints);
      fftInputHost[i].y = 2.0 * moments[i] * std::sin(pi * i / 2.0 / numPoints);
    }
    thrust::device_vector<cufftDoubleComplex> fftInputDevice;
    fftInputDevice = fftInputHost;
    thrust::device_vector<cufftDoubleComplex> fftOutputDevice(numPoints);

    cufftHandle cufftPlan;
    cufftPlan1d(&cufftPlan, numPoints, CUFFT_Z2Z, 1);
    cufftExecZ2Z(cufftPlan, thrust::raw_pointer_cast(fftInputDevice.data()),
                 thrust::raw_pointer_cast(fftOutputDevice.data()), CUFFT_INVERSE);
    thrust::host_vector<cufftDoubleComplex> fftOutputHost;
    fftOutputHost = fftOutputDevice;
    // Extract real part of FFT output as DCT-II coefficients
    std::vector<double> spectralDensity(numPoints, 0.0);
    for (unsigned long int i = 0; i < numPoints / 2; ++i)
    {
      spectralDensity[2 * i] = fftOutputHost[i].x;
      spectralDensity[2 * i + 1] = fftOutputHost[numPoints - i - 1].x;
    }

    // Normalize DCT output to match the spectral density scale
    for (int i = 0; i < numPoints; ++i)
    {
      double x = std::cos(pi * (i + 0.5) / numPoints);
      double weight = 1.0 / (pi * sqrt(1.0 - x * x));
      spectralDensity[i] = spectralDensity[i] * weight;
    }
    // Write spectral density to file
    std::ofstream outFile("spectral_density.dat");
    if (!outFile.is_open())
    {
      throw std::runtime_error("Unable to open file for writing: spectral_density.dat");
    }
    for (int i = 0; i < numPoints; i += 1)
    {
      double energy = std::cos(pi * (i + 0.5) / numPoints);
      outFile << energy * e + c << " " << spectralDensity[i] << "\n";
    }
    outFile.close();

    // Clean up cuFFT and cuBLAS resources
    cufftDestroy(cufftPlan);

    return spectralDensity;
  }

  std::vector<double> computeTypicalDensityOfStates(
      HamiltonianClass &HamiltonianOperator,
      const unsigned long int numMoments,
      const unsigned long int numPoints,
      const double a,
      const double b,
      const unsigned long int numSamples)
  {
    const unsigned long int nSites = HamiltonianOperator.nSites();
    const double pi = M_PI;
    const double c = (a + b) / 2.0;
    const double e = (b - a) / (2.0 - 1e-10);
    // Chebyshev moments
    thrust::device_vector<double> moments(numMoments * numSamples, 0.0);

    // Initialize thrust device vectors for KPM
    thrust::device_vector<double> T0(nSites * numSamples, 0.0);
    thrust::device_vector<double> T1(nSites * numSamples, 0.0);
    thrust::device_vector<double> workspace(nSites * numSamples, 0.0);

    // Compute scaling and shifting parameters for mapping spectrum to [-1, 1]
    double scale = 1.0 / e;
    double shift = -c / e;

    // Generate random initial matrix (numSamples columns)
    thrust::host_vector<double> randomMatrix(nSites * numSamples, 0.0);
    thrust::device_vector<double> randomMatrixDevice;
    thrust::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
    thrust::uniform_int_distribution<unsigned long int> dist(0, nSites);

    for (unsigned long int i = 0; i < numSamples; ++i)
    {
      unsigned long int site = dist(rng);
      randomMatrix[i * nSites + site] = 1.0;
    }
    thrust::copy(randomMatrix.begin(), randomMatrix.end(), T0.begin());

    // Normalize T0 columns using cuBLAS
    thrust::device_vector<double> norms(numSamples, 0.0);
    thrust::device_vector<double> ones(nSites, 1.0);
    thrust::transform(T0.begin(), T0.end(), T0.begin(),
                      [] __device__(double x)
                      { return x * x; });
    double alpha = 1.0, beta = 0.0;
    cublasDgemv(*HamiltonianOperator.cublasHandle(),
                CUBLAS_OP_T,
                nSites,
                numSamples,
                &alpha,
                thrust::raw_pointer_cast(T0.data()),
                nSites,
                thrust::raw_pointer_cast(ones.data()),
                1,
                &beta,
                thrust::raw_pointer_cast(norms.data()),
                1);
    thrust::transform(norms.begin(), norms.end(), norms.begin(),
                      [] __device__(double x)
                      { return 1.0 / sqrt(x); });
    thrust::copy(randomMatrix.begin(), randomMatrix.end(), T0.begin());
    cublasDdgmm(*HamiltonianOperator.cublasHandle(),
                CUBLAS_SIDE_RIGHT,
                nSites,
                numSamples,
                thrust::raw_pointer_cast(T0.data()),
                nSites,
                thrust::raw_pointer_cast(norms.data()),
                1,
                thrust::raw_pointer_cast(T0.data()),
                nSites);
    randomMatrixDevice = T0;
    // Compute T_1 = scaled H * T_0
    HamiltonianOperator.HX(T0, scale, 0.0, shift, T1);
    thrust::fill(moments.begin(), moments.begin() + numSamples, 1.0);
    // Compute moments
    for (unsigned long int m = 1; m < numMoments; ++m)
    {
      thrust::transform(randomMatrixDevice.begin(),
                        randomMatrixDevice.end(),
                        T1.begin(),
                        workspace.begin(),
                        thrust::multiplies<double>());
      cublasDgemv(*HamiltonianOperator.cublasHandle(),
                  CUBLAS_OP_T,
                  nSites,
                  numSamples,
                  &alpha,
                  thrust::raw_pointer_cast(workspace.data()),
                  nSites,
                  thrust::raw_pointer_cast(ones.data()),
                  1,
                  &beta,
                  thrust::raw_pointer_cast(moments.data() + m * numSamples),
                  1);

      if (m < numMoments - 1)
      {
        HamiltonianOperator.HX(T1, 2.0 * scale, -1.0, -2.0 * shift, T0);
        T0.swap(T1);
      }
    }

    // Apply Jackson kernel to moments
    std::vector<double> kernel(numMoments);
    for (unsigned long int m = 0; m < numMoments; ++m)
    {
      kernel[m] = (numMoments - m + 1) * std::cos(pi * m / (numMoments + 1.0)) +
                  std::sin(pi * m / (numMoments + 1.0)) / std::tan(pi / (numMoments + 1.0));
      kernel[m] /= numMoments + 1.0;
    }
    thrust::host_vector<double> momentsHost;
    momentsHost = moments;

    // Use cuFFT to compute DCT-III
    thrust::host_vector<cufftDoubleComplex> fftInputHost(numPoints * numSamples, make_cuDoubleComplex(0.0, 0.0));
    for (unsigned long int i = 0; i < numSamples; ++i)
    {
      fftInputHost[numPoints * i].x = moments[i] * kernel[0];
      for (unsigned long int m = 1; m < numMoments; ++m)
      {
        fftInputHost[numPoints * i + m].x = 2.0 * moments[m * numSamples + i] * kernel[m] * std::cos(pi * m / 2.0 / numPoints);
        fftInputHost[numPoints * i + m].y = 2.0 * moments[m * numSamples + i] * kernel[m] * std::sin(pi * m / 2.0 / numPoints);
      }
    }
    thrust::device_vector<cufftDoubleComplex> fftInputDevice;
    fftInputDevice = fftInputHost;
    thrust::device_vector<cufftDoubleComplex> fftOutputDevice(numPoints * numSamples);

    cufftHandle cufftPlan;
    int rank = 1;
    int n[] = {numPoints};
    int istride = 1, ostride = 1;
    int idist = numPoints, odist = numPoints;
    int batch = numSamples;
    cufftPlanMany(&cufftPlan, rank, n, nullptr, istride, idist,
                  nullptr, ostride, odist, CUFFT_Z2Z, batch);

    cufftExecZ2Z(cufftPlan, thrust::raw_pointer_cast(fftInputDevice.data()),
                 thrust::raw_pointer_cast(fftOutputDevice.data()), CUFFT_INVERSE);
    thrust::host_vector<cufftDoubleComplex> fftOutputHost;
    fftOutputHost = fftOutputDevice;
    // Extract real part of FFT output as DCT-II coefficients
    std::vector<double> spectralDensity(numPoints, 0.0);
    for (unsigned long int i = 0; i < numPoints / 2; ++i)
    {
      for (unsigned long int j = 0; j < numSamples; ++j)
      {
        spectralDensity[2 * i] += std::log(fftOutputHost[numPoints * j + i].x / (pi * sqrt(1.0 - std::cos(pi * (2 * i + 0.5) / numPoints) * std::cos(pi * (2 * i + 0.5) / numPoints))));
        spectralDensity[2 * i + 1] += std::log(fftOutputHost[numPoints * j + numPoints - i - 1].x / (pi * sqrt(1.0 - std::cos(pi * (2 * i + 1.5) / numPoints) * std::cos(pi * (2 * i + 1.5) / numPoints))));
      }
      spectralDensity[2 * i] = std::exp(spectralDensity[2 * i] / numSamples);
      spectralDensity[2 * i + 1] = std::exp(spectralDensity[2 * i + 1] / numSamples);
    }

    // Write spectral density to file
    std::ofstream outFile("spectral_density_typ.dat");
    if (!outFile.is_open())
    {
      throw std::runtime_error("Unable to open file for writing: spectral_density_typ.dat");
    }
    for (int i = 0; i < numPoints; i += 1)
    {
      double energy = std::cos(pi * (i + 0.5) / numPoints);
      outFile << energy * e + c << " " << spectralDensity[i] << "\n";
    }
    outFile.close();

    // Clean up cuFFT and cuBLAS resources
    cufftDestroy(cufftPlan);

    return spectralDensity;
  }
}