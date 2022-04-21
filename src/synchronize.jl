"""
    synchronize!(model; root_rank::Integer = 0)
    synchronize!(ps::Params; root_rank::Integer = 0)
    synchronize!(ps::NamedTuple; root_rank::Integer = 0)

Sync the parameters of the model across all processes.
"""
synchronize!(model; kwargs...) = synchronize!(Flux.params(model); kwargs...)

function synchronize!(ps::Params; root_rank::Integer=0)
    @assert 0 <= root_rank <= total_workers() - 1 "Valid `root_rank` Range: [0, $(total_workers() - 1)]"
    Bcast!.(ps, root_rank, (MPI.COMM_WORLD,))
    return ps
end

function synchronize!(ps::Union{NamedTuple,Tuple}; root_rank::Integer=0)
    @assert 0 <= root_rank <= total_workers() - 1 "Valid `root_rank` Range: [0, $(total_workers() - 1)]"
    return fmap(x -> synchronize!(x; root_rank), ps)
end

function synchronize!(x::AbstractArray{T}; root_rank::Integer=0) where {T<:Number}
    Bcast!(x, root_rank, MPI.COMM_WORLD)
    return x
end

function synchronize!(x::ComponentArray; root_rank::Integer=0)
    d = Bcast!(getdata(x), root_rank, MPI.COMM_WORLD)
    return ComponentArray(d, getaxes(x))
end

# Ideally these things should be Tuples and not arrays
function synchronize!(x::AbstractArray; root_rank::Integer=0)
    synchronize!.(x; root_rank)
end

synchronize!(::Nothing; kwargs...) = nothing

# Synchronizing Symbols should do nothing
synchronize!(s::Symbol; kwargs...) = s

function synchronize!(l::Leaf; root_rank::Integer=0)
    @set! l.state = synchronize!(l.state; root_rank)
end

function synchronize!(x::Number; root_rank::Integer=0)
    return Bcast!([x], root_rank, MPI.COMM_WORLD)[1]
end
