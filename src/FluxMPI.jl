module FluxMPI

include("mpi_extensions.jl")

using .MPIExtensions: Iallreduce!, Ibcast!, JuliaTaskRequest
using CUDA, MPI
using Dates: now
using Flux: params
using Flux.Optimise: AbstractOptimiser
using MPI: Request, Waitall!, Allreduce!, Bcast!
using Zygote: @nograd, Params
import LearnBase: nobs, getobs
import Flux.Optimise: update!, apply!

include("common.jl")
include("synchronize.jl")
include("optimiser.jl")
include("data.jl")

export MPIExtensions, MPI
export local_rank, total_workers, DistributedOptimiser, broadcast_parameters, clean_print, clean_println,
       DistributedDataContainer

end
