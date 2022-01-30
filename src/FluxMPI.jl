module FluxMPI

include("mpi_extensions.jl")

const FluxMPI_initialized = Ref(false)

using .MPIExtensions: Iallreduce!, Ibcast!, JuliaTaskRequest
using CUDA, MPI
using Dates: now
using Flux: params
using Flux.Optimise: AbstractOptimiser
using MPI: Request, Waitall!, Allreduce!
using Zygote: @nograd, Params
import Flux.Optimise: update!, apply!

Initialized() = FluxMPI_initialized[]

"""
    Init(; gpu_devices::Union{Nothing,Vector{Int}} = nothing, verbose::Bool = false)

Setup `FluxMPI`. If GPUs are available and CUDA is functional, each rank is allocated a GPU in a
round-robin fashion.

If calling this function, no need to call `MPI.Init` first.
"""
function Init(; gpu_devices::Union{Nothing,Vector{Int}}=nothing, verbose::Bool=false)
    if Initialized()
        verbose && @warn "FluxMPI already initialized; Skipping..."
        return
    end

    !MPI.Initialized() && MPI.Init()

    rank = local_rank()

    if CUDA.functional()
        gpu_device = if gpu_devices === nothing
            device_count = length(CUDA.devices())
            (rank + 1) % device_count
        else
            gpu_devices[rank + 1]
        end
        verbose && @info "Rank $rank: Using GPU $gpu_device"
        CUDA.device!(gpu_device)
    else
        verbose && @info "Rank $rank: Using CPU"
    end
    FluxMPI_initialized[] = true
    return
end

"""
    local_rank()

Get the rank of the process.
"""
@inline local_rank() = MPI.Comm_rank(MPI.COMM_WORLD)

"""
    total_workers()

Get the total number of workers.
"""
@inline total_workers() = MPI.Comm_size(MPI.COMM_WORLD)

"""
    DistributedOptimiser(optimiser::AbstractOptimiser)

Wrap the `optimiser` in a `DistributedOptimiser`. Before updating the
parameters, this averages the gradients across the processes using
non-blocking Allreduce
"""
struct DistributedOptimiser{B,O<:AbstractOptimiser}
    optimiser::O
    function DistributedOptimiser(optimiser::O; blocking_communication::Bool = false) where {O<:AbstractOptimiser}
        return new{blocking_communication,O}(optimiser)
    end
end

_get_request_type(ps::Params) = _get_request_type(first(ps))
_get_request_type(::CuArray) = JuliaTaskRequest
_get_request_type(::AbstractArray) = Request

function update!(opt::DistributedOptimiser{false}, xs::Params, gs)
    request_count = 0
    requests = Vector{_get_request_type(xs)}(undef, length(xs))
    # Add the gradients across all processes
    s = total_workers()
    for x in xs
        g = gs[x]
        g === nothing && continue
        # Average out the gradients
        g ./= s
        _, request = Iallreduce!(g, +, MPI.COMM_WORLD)
        request_count += 1
        requests[request_count] = request
    end
    # Wait for all the non-blocking operations to be completed
    Waitall!(requests[1:request_count])
    # Update the parameters
    update!(opt.optimiser, xs, gs)
    return
end

function update!(opt::DistributedOptimiser{true}, xs::Params, gs)
    # Add the gradients across all processes
    s = total_workers()
    for x in xs
        g = gs[x]
        g === nothing && continue
        # Average out the gradients
        g ./= s
        Allreduce!(g, +, MPI.COMM_WORLD)
    end
    # Update the parameters
    update!(opt.optimiser, xs, gs)
    return
end

"""
    broadcast_parameters(model; root_rank::Integer = 0)
    broadcast_parameters(ps::Params; root_rank::Integer = 0)

Sync the parameters of the model across all processes.
"""
broadcast_parameters(model; kwargs...) = broadcast_parameters(params(model); kwargs...)

function broadcast_parameters(ps::Params; root_rank::Integer=0)
    @assert 0 <= root_rank <= total_workers() - 1 "Valid `root_rank` Range: [0, $(total_workers() - 1)]"
    requests = Vector{Request}(undef, length(ps))
    for (i, p) in enumerate(ps)
        _, request = Ibcast!(p, root_rank, MPI.COMM_WORLD)
        requests[i] = request
    end
    Waitall!(requests)
    return
end

# TODO: Function to sync optimiser state. Will become easier once Optimisers.jl becomes
#       the standard

"""
    clean_println(args...; kwargs...)

Add `rank` and `size` information to the printed statement
"""
function clean_println(args...; kwargs...)
    rank = local_rank()
    size = total_workers()
    for r in 0:(size - 1)
        r == rank && println("$(now()) [$(rank) / $(size)] ", args...; kwargs...)
        MPI.Barrier(MPI.COMM_WORLD)
    end
    return
end

@nograd clean_println

"""
    clean_print(args...; kwargs...)

Add `rank` and `size` information to the printed statement
"""
function clean_print(args...; kwargs...)
    rank = local_rank()
    size = total_workers()
    for r in 0:(size - 1)
        r == rank && print("$(now()) [$(rank) / $(size)] ", args...; kwargs...)
        MPI.Barrier(MPI.COMM_WORLD)
    end
    return
end

@nograd clean_print

export MPIExtensions, MPI
export local_rank, total_workers, DistributedOptimiser, broadcast_parameters, clean_print, clean_println

end
