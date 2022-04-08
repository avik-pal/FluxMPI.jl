module FluxMPI

include("mpi_extensions.jl")

using CUDA, MPI

import .MPIExtensions: Iallreduce!, Ibcast!, JuliaTaskRequest

import Base: getproperty, setproperty!
import Dates: now
import Flux
import Functors: fmap
import LearnBase: ObsDim
import MLDataUtils: nobs, getobs
import MPI: Request, Waitall!, Allreduce!, Bcast!
import Optimisers: Leaf, init, apply!
import Setfield: @set!
import Zygote: @nograd, Params

# General Utilities -- Init, Clean Printing
include("common.jl")

# synchronize! Model Parameters -- Works with both Zygote.Params which Flux uses and NamedTuples for ExplicitFluxLayers
include("synchronize!.jl")

# Implementation of Distributed Optimiser
include("optimiser.jl")

# Extends LearnBase & MLDataUtils API for Distributed Datasets -- compatible with DataLoaders.jl
include("data.jl")

export MPIExtensions, MPI
export local_rank, total_workers, DistributedOptimiser, clean_print, clean_println,
       DistributedDataContainer

end
