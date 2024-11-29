#include <HamiltonianAnderson.h>
#include <random>
#include <chrono>

namespace kpm
{
  HamiltonianAnderson::HamiltonianAnderson(const unsigned int nDim,
                                           const unsigned long int nSites1D,
                                           const double disorder)
      : d_nDim(nDim),
        d_nSites1D(nSites1D),
        d_nSites((unsigned long int)std::pow(nSites1D, nDim)),
        d_disorder(disorder),
        d_nVec(0)
  {
    CHECK_CUBLAS(cublasCreate(&d_cublasHandle));
    d_cusparsePreprocessDone = false;
    thrust::host_vector<unsigned long int> rowOffsets(d_nSites + 1, 0);
    thrust::host_vector<unsigned long int> colIndex;
    thrust::host_vector<double> HamiltonianData;

    thrust::host_vector<double> randomPotential(d_nSites);
    thrust::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
    thrust::uniform_real_distribution<double> dist(-d_disorder / 2.0, d_disorder / 2.0);
    thrust::generate(randomPotential.begin(), randomPotential.end(), [&]
                     { return dist(rng); });

    // Helper function to compute the 1D index given multi-dimensional coordinates
    auto getSiteIndex = [&](std::vector<unsigned long int> &coords)
    {
      unsigned long int index = 0;
      for (unsigned int dim = 0; dim < d_nDim; ++dim)
      {
        index += coords[dim] * std::pow(d_nSites1D, dim);
      }
      return index;
    };

    unsigned long int nnz = 0; // Number of non-zero entries
    for (unsigned long int site = 0; site < d_nSites; ++site)
    {
      // Add diagonal term (on-site disorder potential)
      colIndex.push_back(site);
      HamiltonianData.push_back(randomPotential[site]);
      ++nnz;

      // Compute the multi-dimensional coordinates of the current site
      std::vector<unsigned long int> coords(d_nDim);
      unsigned long int tempSite = site;
      for (unsigned int dim = 0; dim < d_nDim; ++dim)
      {
        coords[dim] = tempSite % d_nSites1D;
        tempSite /= d_nSites1D;
      }

      // Add hopping to neighbors in each dimension (periodic boundary)
      for (unsigned int dim = 0; dim < d_nDim; ++dim)
      {
        for (int shift : {-1, 1})
        { // Neighbors in both directions
          // Create a copy of the current coordinates
          std::vector<unsigned long int> neighborCoords = coords;
          neighborCoords[dim] = (neighborCoords[dim] + shift + d_nSites1D) % d_nSites1D; // Apply PBC

          // Get the index of the neighboring site
          unsigned long int neighbor = getSiteIndex(neighborCoords);

          colIndex.push_back(neighbor);
          HamiltonianData.push_back(-1.0); // Hopping amplitude
          ++nnz;
        }
      }
      // Update row offsets
      rowOffsets[site + 1] = nnz;
    }
    // Copy CSR arrays to device
    d_HamiltonianrowOffsets = rowOffsets;
    d_HamiltoniancolIndices = colIndex;
    d_HamiltonianData = HamiltonianData;

    CHECK_CUSPARSE(cusparseCreate(&d_cusparseHandle));
    // Create cuSPARSE matrix descriptor in column-major order
    // cusparseCreateCsr expects row indices and column indices swapped for column-major format
    CHECK_CUSPARSE(cusparseCreateCsr(
        &d_sparseHamiltonianDescr,
        d_nSites,
        d_nSites,
        nnz,
        thrust::raw_pointer_cast(d_HamiltonianrowOffsets.data()),
        thrust::raw_pointer_cast(d_HamiltoniancolIndices.data()),
        thrust::raw_pointer_cast(d_HamiltonianData.data()),
        CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    // // Print the Hamiltonian in dense format
    // std::cout << "Hamiltonian (Dense format):" << std::endl;
    // for (unsigned long int i = 0; i < d_nSites; ++i)
    // {
    //   for (unsigned long int j = 0; j < d_nSites; ++j)
    //   {
    //     // Find if (i, j) is in the sparse matrix
    //     double value = 0.0;
    //     for (unsigned long int k = rowOffsets[i]; k < rowOffsets[i + 1]; ++k)
    //     {
    //       if (colIndex[k] == j)
    //       {
    //         value = HamiltonianData[k];
    //         break;
    //       }
    //     }
    //     std::cout << value << "\t";
    //   }
    //   std::cout << std::endl;
    // }
  }

  HamiltonianAnderson::~HamiltonianAnderson()
  {
    CHECK_CUSPARSE(cusparseDestroy(d_cusparseHandle));
    CHECK_CUBLAS(cublasDestroy(d_cublasHandle));
  }

  void HamiltonianAnderson::HX(thrust::device_vector<double> &src,
                               const double scalarHX,
                               const double scalarY,
                               const double scalarX,
                               thrust::device_vector<double> &dst)
  {
    // Ensure src and dst sizes are compatible with the number of sites
    assert(src.size() % d_nSites == 0);
    assert(dst.size() % d_nSites == 0);
    assert(src.size() == dst.size());

    // Number of vectors
    const unsigned int nVec = src.size() / d_nSites;
    const double scalarOne = 1.0;

    // Preprocessing: Create dense matrix descriptors and compute buffer if needed
    // Create dense matrix descriptors for src and dst
    CHECK_CUSPARSE(cusparseCreateDnMat(&d_srcDescr, d_nSites, nVec, d_nSites,
                                       thrust::raw_pointer_cast(src.data()),
                                       CUDA_R_64F, CUSPARSE_ORDER_COL));

    CHECK_CUSPARSE(cusparseCreateDnMat(&d_dstDescr, d_nSites, nVec, d_nSites,
                                       thrust::raw_pointer_cast(dst.data()),
                                       CUDA_R_64F, CUSPARSE_ORDER_COL));

    if (!d_cusparsePreprocessDone || d_nVec != nVec)
    {
      // Compute buffer size
      size_t bufferSize = 0;
      CHECK_CUSPARSE(cusparseSpMM_bufferSize(
          d_cusparseHandle,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &scalarHX,
          d_sparseHamiltonianDescr,
          d_srcDescr,
          &scalarOne,
          d_dstDescr,
          CUDA_R_64F,
          CUSPARSE_SPMM_ALG_DEFAULT,
          &bufferSize));

      // Allocate buffer
      d_cusparseBuffer.resize(bufferSize);
      d_nVec = nVec;
      d_cusparsePreprocessDone = true;
    }
    // Update the dense matrix descriptors with the current pointers for src and dst
    CHECK_CUSPARSE(cusparseDnMatSetValues(d_srcDescr, thrust::raw_pointer_cast(src.data())));
    CHECK_CUSPARSE(cusparseDnMatSetValues(d_dstDescr, thrust::raw_pointer_cast(dst.data())));

    // Step 1: dst = scalarY * dst (in place) using cuBLAS
    double *dst_ptr = thrust::raw_pointer_cast(dst.data());
    CHECK_CUBLAS(cublasDscal(
        d_cublasHandle,
        dst.size(),
        &scalarY,
        dst_ptr,
        1));

    // Step 2: dst += scalarX * src (in place) using cuBLAS
    const double *src_ptr = thrust::raw_pointer_cast(src.data());
    CHECK_CUBLAS(cublasDaxpy(
        d_cublasHandle,
        dst.size(),
        &scalarX,
        src_ptr,
        1,
        dst_ptr,
        1));

    // Step 3: dst += scalarHX * (Hamiltonian * src) using cuSPARSE
    CHECK_CUSPARSE(cusparseSpMM(
        d_cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &scalarHX,
        d_sparseHamiltonianDescr,
        d_srcDescr,
        &scalarOne,
        d_dstDescr,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        thrust::raw_pointer_cast(d_cusparseBuffer.data())));
  }

} // namespace kpm
