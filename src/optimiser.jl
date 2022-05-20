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
