"""
    DistributedOptimiser(optimiser)

Wrap the `optimiser` in a `DistributedOptimiser`. Before updating the
parameters, this adds the gradients across the processes using
non-blocking Allreduce

!!! note
    Remember to scale the loss function by `1 / total_workers()` to ensure
    that the gradients are correctly averaged
"""
struct DistributedOptimiser{O}
    optimiser::O
end

function apply!(o::DistributedOptimiser, state, x, y)
    y_ = allreduce!(y, +, MPI.COMM_WORLD)
    apply!(o.optimiser, state, x, y_)
end

init(o::DistributedOptimiser, x::AbstractArray) = init(o.optimiser, x)


"""
    allreduce_gradients(gs::NamedTuple; on_gpu::Bool=CUDA.functional())
"""
function allreduce_gradients(gs::NamedTuple; on_gpu::Bool=CUDA.functional())
    if on_gpu
        # Transfer data to CPU since OpenMPI Iallreduce! doesn't work for CUDA
        gs = fmap(cpu, gs)
    end
    requests = MPI.Request[]
    function nonblocking_reduce_gradients(g)
        g, req = Iallreduce!(g, +, MPI.COMM_WORLD)
        push!(requests, req)
        return g
    end
    gs = fmap(nonblocking_reduce_gradients, gs)
    MPI.Waitall!(requests)
    if on_gpu
        # Transfer data back to GPU
        gs = fmap(gpu, gs)
    end
    return gs
end
