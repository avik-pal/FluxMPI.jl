module FluxMPI

include("mpi_extensions.jl")

# General Utilities -- Init, Clean Printing
include("common.jl")

# synchronize!
include("synchronize.jl")

# Implementation of Distributed Optimiser
include("optimiser.jl")

# Support for MLUtils.jl DataLoader
include("data.jl")

export local_rank, total_workers, DistributedOptimizer, clean_print, clean_println,
       DistributedDataContainer, allreduce_gradients

end
