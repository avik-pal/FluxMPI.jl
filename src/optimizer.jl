"""
    DistributedOptimizer(optimizer)

Wrap the `optimizer` in a `DistributedOptimizer`. Before updating the parameters, this adds
the gradients across the processes using non-blocking Allreduce

## Arguments

  - `optimizer`: An Optimizer compatible with the Optimisers.jl package

!!! note

    Remember to scale the loss function by `1 / total_workers()` to ensure
    that the gradients are correctly averaged
"""
struct DistributedOptimizer{O} <: Optimisers.AbstractRule
  optimizer::O
end

function Optimisers.apply!(o::DistributedOptimizer, state, x, y)
  y_ = allreduce!(y, +, MPI.COMM_WORLD)
  return Optimisers.apply!(o.optimizer, state, x, y_)
end

Optimisers.init(o::DistributedOptimizer, x::AbstractArray) = Optimisers.init(o.optimizer, x)

"""
    allreduce_gradients(gs::NamedTuple; on_gpu::Bool=CUDA.functional())

Allreduce the gradients. This uses a non-blocking API which will be efficient for large
containers of multiple parameter arrays.

## Arguments

  - `gs`: A `NamedTuple` of gradients

## Keyword Arguments

  - `on_gpu`: Specify if the gradients are on GPU. Defaults to `CUDA.functional()`

## Returns

  - `Allreduce`d NamedTuple of gradients
"""
function allreduce_gradients(gs::NamedTuple; on_gpu::Bool=CUDA.functional())
  # Transfer data to CPU since OpenMPI Iallreduce! doesn't work for CUDA
  gs = on_gpu ? fmap(cpu, gs) : gs

  requests = MPI.Request[]

  nonblocking_reduce_gradients(g) = g
  function nonblocking_reduce_gradients(g::AbstractArray)
    g, req = Iallreduce!(g, +, MPI.COMM_WORLD)
    push!(requests, req)
    return g
  end

  gs = fmap(nonblocking_reduce_gradients, gs)
  MPI.Waitall!(requests)

  # Transfer data back to GPU
  gs = on_gpu ? fmap(gpu, gs) : gs

  return gs
end
