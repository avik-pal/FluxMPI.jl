"""
    DistributedOptimiser(optimiser)

Wrap the `optimiser` in a `DistributedOptimiser`. Before updating the
parameters, this averages the gradients across the processes using
non-blocking Allreduce
"""
struct DistributedOptimiser{O}
    optimiser::O
end

function getproperty(opt::DistributedOptimiser, name::Symbol)
    if name == :optimiser
        return getfield(opt, :optimiser)
    else
        return getproperty(opt.optimiser, name)
    end
end

function setproperty!(opt::DistributedOptimiser, name::Symbol, x::Any)
    if name == :optimiser
        error("Cannot update `optimiser` property of `DistributedOptimiser`")
    else
        setproperty!(opt.optimiser, name, x)
    end
end

function apply!(o::DistributedOptimiser{true}, state, x, x̄)
    # Average gradients across all processes
    x̄ ./= total_workers()
    Allreduce!(x̄, +, MPI.COMM_WORLD)
    apply!(o.optimiser, state, x, x̄)
end

init(o::DistributedOptimiser, x::AbstractArray) = init(o.optimiser, x)
