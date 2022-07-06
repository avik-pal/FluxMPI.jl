import ChainRulesCore as CRC
import CUDA, Dates, MPI

const FluxMPI_initialized = Ref(false)

"""
    Initialized()

Has FluxMPI been initialized?
"""
Initialized() = FluxMPI_initialized[]

"""
    Init(; gpu_devices::Union{Nothing,Vector{Int}} = nothing, verbose::Bool = false)

Setup `FluxMPI`. If GPUs are available and CUDA is functional, each rank is allocated a
GPU in a round-robin fashion.

If calling this function, no need to call `MPI.Init` first.
"""
function Init(; gpu_devices::Union{Nothing, Vector{Int}}=nothing, verbose::Bool=false)
  if Initialized()
    verbose && clean_println("FluxMPI already initialized; Skipping...")
    return
  end

  !MPI.Initialized() && MPI.Init()
  FluxMPI_initialized[] = true

  if verbose && total_workers() == 1
    @warn "Using FluxMPI with only 1 worker. It might be faster to run the code without MPI" maxlog=1
  end

  rank = local_rank()

  if CUDA.functional()
    gpu_device = if gpu_devices === nothing
      device_count = length(CUDA.devices())
      (rank + 1) % device_count
    else
      gpu_devices[rank + 1]
    end
    verbose && clean_println("Using GPU $gpu_device")
    CUDA.device!(gpu_device)
  else
    verbose && clean_println("Using CPU")
  end

  return
end

"""
    local_rank()

Get the rank of the process.
"""
@inline function local_rank()
  !Initialized() && error("FluxMPI has not been initialized")
  return MPI.Comm_rank(MPI.COMM_WORLD)
end

CRC.@non_differentiable local_rank()

"""
    total_workers()

Get the total number of workers.
"""
@inline function total_workers()
  !Initialized() && error("FluxMPI has not been initialized")
  return MPI.Comm_size(MPI.COMM_WORLD)
end

CRC.@non_differentiable total_workers()

# Print Functions
for print_fn in (:println, :print)
  function_name = Symbol("clean_" * string(print_fn))
  @eval begin
    function $(function_name)(args...; kwargs...)
      if !Initialized()
        $(print_fn)("$(Dates.now()) ", args...; kwargs...)
        return
      end
      rank = local_rank()
      size = total_workers()
      if size == 1
        $(print_fn)(args...; kwargs...)
        return
      end
      for r in 0:(size - 1)
        if r == rank
          $(print_fn)("$(now()) [$(rank) / $(size)] ", args...; kwargs...)
          flush(stdout)
        end
        MPI.Barrier(MPI.COMM_WORLD)
      end
      return
    end

    CRC.@non_differentiable $(function_name)(::Any...)
  end
end

"""
    clean_println(args...; kwargs...)

Add `rank` and `size` information to the printed statement
"""
clean_println

"""
    clean_print(args...; kwargs...)

Add `rank` and `size` information to the printed statement
"""
clean_print
