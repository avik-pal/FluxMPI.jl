"""
    DistributedOptimiser(optimiser::AbstractOptimiser)

Wrap the `optimiser` in a `DistributedOptimiser`. Before updating the
parameters, this averages the gradients across the processes using
non-blocking Allreduce
"""
struct DistributedOptimiser{B,O<:AbstractOptimiser}
    optimiser::O
    # TODO: Check if non-blocking calls are better. For small scale problems blocking calls are faster
    #       but that could be because we are unable to use Iallreduce with CuArrays
    function DistributedOptimiser(optimiser::O; blocking_communication::Bool=true) where {O<:AbstractOptimiser}
        return new{blocking_communication,O}(optimiser)
    end
end

function getproperty(opt::DistributedOptimiser, name::Symbol)
    if name == :optimiser || name == :optimizer
        return getfield(opt, :optimiser)
    else
        return getproperty(opt.optimiser, name)
    end
end

function setproperty!(opt::DistributedOptimiser, name::Symbol, x::Any)
    if name == :optimiser || name == :optimizer
        error("Cannot update `optimiser` property of `DistributedOptimiser`")
    else
        setproperty!(opt.optimiser, name, x)
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