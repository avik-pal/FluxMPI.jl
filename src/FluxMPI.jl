module FluxMPI

using Flux, CUDA, MPI, Zygote, Random
using Random: AbstractRNG, shuffle!, GLOBAL_RNG

const mpi_is_cuda_aware = Ref(false)
const FluxMPI_initialized = Ref(false)

function __init__()
    # If using `mpi_is_cuda_aware` anywhere use the @static macro
    # since we anyways need to recompile the code when MPI
    # implementation changes
    mpi_is_cuda_aware[] = MPI.has_cuda()
    if !mpi_is_cuda_aware[]
        @warn "MPI Implementation is not CUDA Aware"
    end
end

Initialized() = FluxMPI_initialized[]

function Init(; gpu_devices::Union{Nothing,Vector{Int}} = nothing)
    if Initialized()
        @warn "FluxMPI already initialized; Skipping..."
        return
    end

    !MPI.Initialized() && MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if CUDA.functional()
        gpu_device = if gpu_devices === nothing
            device_count = length(CUDA.devices())
            (rank + 1) % device_count
        else
            gpu_devices[rank+1]
        end
        @info "Rank $rank: Using GPU $gpu_device"
        CUDA.device!(gpu_device)
    else
        @info "Rank $rank: Using CPU"
    end
    FluxMPI_initialized[] = true
    return
end

struct DataParallelFluxModel{D,M}
    model::M
end

struct DataParallelParamsWrapper{D,B,S,L,P}
    buffer::B
    sizes::S
    lengths::L
    params::P
end

Flux.@functor DataParallelFluxModel

Flux.trainable(dp::DataParallelFluxModel) = Flux.trainable(dp.model)

function Flux.params(dp::DataParallelFluxModel{D}) where {D}
    ps = Flux.params(dp.model)
    _buffer = zero.(ps)
    sizes = size.(_buffer)
    lengths = length.(_buffer)
    buffer = vcat(vec.(_buffer)...)
    return DataParallelParamsWrapper{
        D,
        typeof(buffer),
        typeof(sizes),
        typeof(lengths),
        typeof(ps),
    }(
        buffer,
        sizes,
        lengths,
        ps,
    )
end

Zygote.Params(ps::DataParallelParamsWrapper) = ps

Flux.Optimise.update!(opt, xs::DataParallelParamsWrapper, gs) =
    Flux.Optimise.update!(opt, xs.params, gs)

function DataParallelFluxModel(model, gpu_devices::Union{Nothing,Vector{Int}} = nothing)
    if !Initialized()
        @warn "FluxMPI not initialised, initialising now"
        Init(; gpu_devices = gpu_devices)
    end

    comm = MPI.COMM_WORLD
    comm_size = MPI.Comm_size(comm)

    p, re = Flux.destructure(model)
    safe_bcast!(p, 0, MPI.COMM_WORLD)
    model = re(p)

    if CUDA.functional()
        model = model |> gpu
    end

    return DataParallelFluxModel{Val(comm_size),typeof(model)}(model)
end

(dp::DataParallelFluxModel)(args...) = dp.model(args...)


function flatten_grads(ps::DataParallelParamsWrapper, gs::Zygote.Grads)
    idx = 1
    for (i, p) in enumerate(gs.params)
        l = ps.lengths[i]
        ps.buffer[idx:idx+l-1] .= vec(gs[p])
        idx += l
    end
    return ps.buffer
end

function unflatten_grads!(
    gs::Zygote.Grads,
    ps::DataParallelParamsWrapper,
    gs_flattened::AbstractArray,
)
    idx = 1
    for (i, p) in enumerate(gs.params)
        l = ps.lengths[i]
        gs[p] = reshape(gs_flattened[idx:idx+l-1], ps.sizes[i])
        idx += l
    end
    return gs
end


function Zygote.withgradient(func, ps::DataParallelParamsWrapper)
    comm = MPI.COMM_WORLD
    size = MPI.Comm_size(comm)

    y, back = Zygote.pullback(func, ps.params)
    gs = back(Zygote.sensitivity(y))

    gs_flattened = flatten_grads(ps, gs) ./ size

    gs_flattened = safe_allreduce!(gs_flattened, +, comm)

    return (val = y, grad = unflatten_grads!(gs, ps, gs_flattened))
end

Zygote.withgradient(func, ps::DataParallelParamsWrapper{Val{1}}) =
    Zygote.withgradient(func, ps.params)

Zygote.gradient(func, ps::DataParallelParamsWrapper) =
    Zygote.withgradient(func, ps).grad


function DataParallelDataLoader(
    data;
    batchsize = 1,
    shuffle = false,
    partial = true,
    rng = GLOBAL_RNG,
)
    batchsize > 0 || throw(ArgumentError("Need positive batchsize"))

    comm = MPI.COMM_WORLD
    size = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    _n = Flux.Data._nobs(data)

    partitions = collect(
        Iterators.partition(Random.shuffle(rng, 1:_n), Int(ceil(_n / size))),
    )
    idxs = collect(partitions[rank+1])

    n = length(idxs)

    if n < batchsize
        @warn "Number of observations less than batchsize, decreasing the batchsize to $n"
        batchsize = n
    end

    imax = partial ? n : n - batchsize + 1
    return Flux.Data.DataLoader(
        data,
        batchsize,
        n,
        partial,
        imax,
        idxs,
        shuffle,
        rng,
    )
end


function safe_allreduce!(v::AbstractArray, op, comm)
    MPI.Allreduce!(v, op, comm)
    return v
end

function safe_allreduce!(v::CuArray, op, comm)
    @static if !mpi_is_cuda_aware[]
        # Do transfer on CPU since MPI is not CUDA aware
        v = v |> cpu
    end
    MPI.Allreduce!(v, op, comm)
    @static if !mpi_is_cuda_aware[]
        v = v |> gpu
    end
    return v
end

function safe_bcast!(v::AbstractArray, root::Integer, comm)
    MPI.Bcast!(v, root, comm)
    return v
end

function safe_bcast!(v::CuArray, root::Integer, comm)
    @static if !mpi_is_cuda_aware[]
        # Do transfer on CPU since MPI is not CUDA aware
        v = v |> cpu
    end
    MPI.Bcast!(v, root, comm)
    @static if !mpi_is_cuda_aware[]
        v = v |> gpu
    end
    return v
end

function safe_reduce!(v::AbstractArray, op, root::Integer, comm)
    MPI.Reduce!(v, op, root, comm)
    return v
end

function safe_reduce!(v::CuArray, op, root::Integer, comm)
    @static if !mpi_is_cuda_aware[]
        # Do transfer on CPU since MPI is not CUDA aware
        v = v |> cpu
    end
    MPI.Reduce!(v, op, root, comm)
    @static if !mpi_is_cuda_aware[]
        v = v |> gpu
    end
    return v
end


export DataParallelFluxModel, DataParallelDataLoader
export safe_allreduce!, safe_bcast!, safe_reduce!

end