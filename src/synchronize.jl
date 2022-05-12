"""
    synchronize!(x::NamedTuple; root_rank::Integer = 0)
    synchronize!(x::Tuple; root_rank::Integer = 0)
    synchronize!(x::AbstractArray; root_rank::Integer = 0)
    synchronize!(x::ComponentArray; root_rank::Integer = 0)
    synchronize!(x::Nothing; root_rank::Integer = 0)
    synchronize!(x::Symbol; root_rank::Integer = 0)
    synchronize!(x::Leaf; root_rank::Integer = 0)
    synchronize!(x::Number; root_rank::Integer = 0)

Synchronize `x` across all processes.
"""
function synchronize!(ps::Union{NamedTuple,Tuple}; root_rank::Integer=0)
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

function synchronize!(l::Leaf; root_rank::Integer=0)
    @set! l.state = synchronize!(l.state; root_rank)
end

function synchronize!(x::Number; root_rank::Integer=0)
    return Bcast!([x], root_rank, MPI.COMM_WORLD)[1]
end

# If we don't know what to synchronize, we don't do it
# -- Symbols, Nothing, Missing, Val, etc.
synchronize!(x; kwargs...) = x