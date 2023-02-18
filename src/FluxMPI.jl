module FluxMPI

using Adapt, CUDA, Dates, Functors, MPI, Optimisers, Setfield
using Preferences
import ChainRulesCore as CRC

const FluxMPI_Initialized = Ref(false)

# Extensions
if !isdefined(Base, :get_extension)
  using Requires
end

const MPI_IS_CUDA_AWARE = Ref(false)

function __init__()
  if haskey(ENV, "FLUXMPI_DISABLE_CUDAMPI_SUPPORT")
    @warn "FLUXMPI_DISABLE_CUDAMPI_SUPPORT environment variable has been removed and has no effect. Please use `FluxMPI.disable_cudampi_support()` instead." maxlog=1``
  end

  disable_cuda_aware_support = @load_preference("FluxMPIDisableCUDAMPISupport", false)

  MPI_IS_CUDA_AWARE[] = !disable_cuda_aware_support && MPI.has_cuda()

  if disable_cuda_aware_support && MPI.has_cuda()
    @info "CUDA-aware MPI support disabled using LocalPreferences.toml." maxlog=1
  elseif !MPI_IS_CUDA_AWARE[]
    @warn "MPI Implementation is not CUDA Aware." maxlog=1
  else
    @info "MPI Implementation is CUDA Aware." maxlog=1
  end

  @static if !isdefined(Base, :get_extension)
    # Handling ComponentArrays
    @require ComponentArrays="b0b7db55-cfe3-40fc-9ded-d10e2dbeff66" begin include("../ext/FluxMPIComponentArraysExt.jl") end

    # Handling Flux
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin include("../ext/FluxMPIFluxExt.jl") end
  end
end

"""
    disable_cudampi_support(; disable=true)

Disable CUDA MPI support. Julia Session needs to be restarted for this to take effect.
"""
function disable_cudampi_support(; disable=true)
  @set_preferences!("FluxMPIDisableCUDAMPISupport"=>disable)
  @info "CUDA-aware MPI support "*
        (disable ? "disabled" : "enabled")*
        ". Restart Julia for this change to take effect!" maxlog=1
end

# Error Handling
struct FluxMPINotInitializedError <: Exception end

function Base.showerror(io::IO, e::FluxMPINotInitializedError)
  return print(io, "Please call FluxMPI.init(...) before using FluxMPI functionalities!")
end

# MPI Extensions
include("mpi_extensions.jl")

# General Utilities -- Init, Clean Printing
include("common.jl")

# synchronize!
include("synchronize.jl")

# Implementation of Distributed Optimiser
include("optimizer.jl")

# Support for MLUtils.jl DataLoader
include("data.jl")

# Extensions: Flux
## We need this because Flux works on arbitrary types. Which means we cannot dispatch
## `synchronize!` correctly. It is the user's responsibility to call `synchronize!` on
## `FluxMPIFluxModel`.
struct FluxMPIFluxModel{M}
  model::M
end

export local_rank, total_workers, DistributedOptimizer, fluxmpi_print, fluxmpi_println,
       DistributedDataContainer, allreduce_gradients

export FluxMPIFluxModel

end
