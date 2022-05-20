module FluxMPI

include("mpi_extensions.jl")

import ChainRulesCore: @non_differentiable
import ComponentArrays: ComponentArray, getdata, getaxes
import CUDA
import Dates: now
import Functors: fmap
import LearnBase
import MLDataUtils
import MLUtils: getobs, numobs
import MPI
import MPI: Barrier, Comm_rank, Comm_size, COMM_WORLD
import .MPIExtensions: allreduce!, bcast!, reduce!, Iallreduce!, Ibcast!
import Optimisers: Leaf, init, apply!
import Setfield: @set!

# General Utilities -- Init, Clean Printing
include("common.jl")

# synchronize! Model Parameters -- Works with both Zygote.Params which Flux uses and NamedTuples for ExplicitFluxLayers
include("synchronize.jl")

# Implementation of Distributed Optimiser
include("optimiser.jl")

# Extends LearnBase & MLDataUtils API for Distributed Datasets -- compatible with DataLoaders.jl
include("data.jl")

export local_rank, total_workers, DistributedOptimiser, clean_print, clean_println, DistributedDataContainer

end
