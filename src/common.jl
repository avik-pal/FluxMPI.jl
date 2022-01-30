const FluxMPI_initialized = Ref(false)

"""
    Initialized()

Has FluxMPI been initialized?
"""
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