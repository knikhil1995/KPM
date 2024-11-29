#include <Hamiltonian.h>
#include <HamiltonianAnderson.h>
#include <headers.h>
#include <lanczos.h>
#include <moments.h>
int main(int argc, char* argv[])
{
  std::shared_ptr<kpm::HamiltonianClass> HamiltonianPtr;
  HamiltonianPtr = std::make_shared<kpm::HamiltonianAnderson>(3, 50, std::atof(argv[1]));
  auto eigenBounds = lanczosLowerUpperBoundEigenSpectrum(*HamiltonianPtr);
  computeTotalDensityOfStates(
      *HamiltonianPtr,
      2000,
      4000,
      eigenBounds.first,
      eigenBounds.second,
      100);
  computeTypicalDensityOfStates(
      *HamiltonianPtr,
      2000,
      4000,
      eigenBounds.first,
      eigenBounds.second,
      100);
  return 0;
}
