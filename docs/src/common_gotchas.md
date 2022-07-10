## CUDA-aware MPI

### Setup

OpenMPI has extensive instructions on building
[CUDA-aware MPI](https://www-lb.open-mpi.org/faq/?category=buildcuda). Next rebuild MPI.jl
using these
[instructions](https://juliaparallel.org/MPI.jl/stable/configuration/#Using-a-system-provided-MPI).

Additionally, make sure to set `JULIA_CUDA_USE_MEMPOOL=none`.

### Should you use CUDA-aware MPI?

I would recommend **not** using this atm, since `JULIA_CUDA_USE_MEMPOOL=none` will severely
slow down your code (*~2-3x* for most workloads I tested). Instead setup `MPI.jl` using you
system provided MPI and set `FLUXMPI_DISABLE_CUDAMPI_SUPPORT=true`.
