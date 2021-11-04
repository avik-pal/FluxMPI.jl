module FluxMPI

using Flux, CUDA, MPI, Zygote

const mpi_is_cuda_aware = Ref(false)

function __init__()
    mpi_is_cuda_aware[] = MPI.has_cuda()
end

struct DataParallelFluxModel{M}
    model::M
    device_count::Int
end

struct DataParallelParamsWrapper{B,S,L,P}
    buffer::B
    sizes::S
    lengths::L
    params::P
end

Flux.@functor DataParallelFluxModel

Flux.trainable(dp::DataParallelFluxModel) = Flux.trainable(dp.model)

function Flux.params(dp::DataParallelFluxModel)
    ps = Flux.params(dp.model)
    _buffer = zero.(ps)
    sizes = size.(_buffer)
    lengths = length.(_buffer)
    buffer = vcat(vec.(_buffer)...)
    return DataParallelParamsWrapper(buffer, sizes, lengths, ps)
end

function DataParallelFluxModel(model, gpu_devices::Vector{Int} = [])
    device_count = length(gpu_devices)

    if CUDA.functional()
        @assert device_count > 0

        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)

        gpu_id = gpu_devices[rank+1]

        @info "Rank $rank: Using GPU $gpu_id"
        CUDA.device!(gpu_id)

        model = model |> gpu
    end

    return DataParallelFluxModel(model, device_count)
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


function Zygote.gradient(func, ps::DataParallelParamsWrapper)
    comm = MPI.COMM_WORLD
    size = MPI.Comm_size(comm)

    gs = Zygote.gradient(func, ps.params)

    gs_flattened = flatten_grads(ps, gs)

    if !mpi_is_cuda_aware[]
        # Do transfer on CPU since MPI is not CUDA aware
        gs_flattened = gs_flattened |> cpu
    end

    MPI.Allreduce!(gs_flattened, +, comm)

    if CUDA.functional() && !mpi_is_cuda_aware[]
        # If CUDA is functional the final result should be on GPU
        # If MPI were CUDA aware we never transfered the data to CPU
        gs_flattened = gs_flattened |> gpu
    end

    return unflatten_grads!(gs, ps, gs_flattened ./ size)
end


export DataParallelFluxModel

end