#include <headers.h>
#include <Hamiltonian.h>
namespace kpm
{
  std::vector<double> computeTotalDensityOfStates(
      HamiltonianClass &HamiltonianOperator,
      const unsigned long int numMoments,
      const unsigned long int numPoints,
      const double a,
      const double b,
      const unsigned long int numSamples);
  std::vector<double> computeTypicalDensityOfStates(
      HamiltonianClass &HamiltonianOperator,
      const unsigned long int numMoments,
      const unsigned long int numPoints,
      const double a,
      const double b,
      const unsigned long int numSamples);
}