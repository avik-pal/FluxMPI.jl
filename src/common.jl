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
        verbose && clean_println("FluxMPI already initialized; Skipping...")
        return
    end

    !MPI.Initialized() && MPI.Init()
    FluxMPI_initialized[] = true

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
    return Comm_rank(COMM_WORLD)
end

@non_differentiable local_rank()

"""
    total_workers()

Get the total number of workers.
"""
@inline function total_workers()
    !Initialized() && error("FluxMPI has not been initialized")
    return Comm_size(COMM_WORLD)
end

@non_differentiable total_workers()

"""
    clean_println(args...; kwargs...)

Add `rank` and `size` information to the printed statement
"""
function clean_println(args...; kwargs...)
    if !Initialized()
        println("$(now()) ", args...; kwargs...)
        return
    end
    rank = local_rank()
    size = total_workers()
    if size == 1
        println(args...; kwargs...)
        return
    end
    for r in 0:(size - 1)
        if r == rank
            println("$(now()) [$(rank) / $(size)] ", args...; kwargs...)
            flush(stdout)
        end
        Barrier(COMM_WORLD)
    end
    return
end

@non_differentiable clean_println(::Any...)

"""
    clean_print(args...; kwargs...)

Add `rank` and `size` information to the printed statement
"""
function clean_print(args...; kwargs...)
    if !Initialized()
        print("$(now()) ", args...; kwargs...)
        return
    end
    rank = local_rank()
    size = total_workers()
    if size == 1
        print(args...; kwargs...)
        return
    end
    for r in 0:(size - 1)
        if r == rank
            print("$(now()) [$(rank) / $(size)] ", args...; kwargs...)
            flush(stdout)
        end
        Barrier(COMM_WORLD)
    end
    return
end

@non_differentiable clean_print(::Any...)
