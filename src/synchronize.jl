# TODO: Function to sync optimiser state. Will become easier once Optimisers.jl becomes
#       the standard

"""
broadcast_parameters(model; root_rank::Integer = 0, blocking_communication::Bool = true)
broadcast_parameters(ps::Params; root_rank::Integer = 0, blocking_communication::Bool = true)

Sync the parameters of the model across all processes.
"""
broadcast_parameters(model; kwargs...) = broadcast_parameters(params(model); kwargs...)

function broadcast_parameters(ps::Params; root_rank::Integer=0, blocking_communication::Bool=true)
    @assert 0 <= root_rank <= total_workers() - 1 "Valid `root_rank` Range: [0, $(total_workers() - 1)]"
    if blocking_communication
        requests = Vector{Request}(undef, length(ps))
        for (i, p) in enumerate(ps)
            _, request = Ibcast!(p, root_rank, MPI.COMM_WORLD)
            requests[i] = request
        end
        Waitall!(requests)
    else
        Bcast!.(ps, root_rank, MPI.COMM_WORLD)
    end
    return
end