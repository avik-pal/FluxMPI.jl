import ComponentArrays, Functors, MPI, Optimisers, Setfield
import .MPIExtensions

"""
    synchronize!(x::NamedTuple; root_rank::Integer=0)
    synchronize!(x::Tuple; root_rank::Integer=0)
    synchronize!(x::AbstractArray; root_rank::Integer=0)
    synchronize!(x::ComponentArray; root_rank::Integer=0)
    synchronize!(x::Nothing; root_rank::Integer=0)
    synchronize!(x::Symbol; root_rank::Integer=0)
    synchronize!(x::Leaf; root_rank::Integer=0)
    synchronize!(x::Number; root_rank::Integer=0)

Synchronize `x` across all processes.
"""
function synchronize!(ps::Union{NamedTuple, Tuple}; root_rank::Integer=0)
  return Functors.fmap(x -> synchronize!(x; root_rank), ps)
end

function synchronize!(x::AbstractArray{T}; root_rank::Integer=0) where {T <: Number}
  MPIExtensions.bcast!(x, root_rank, MPI.COMM_WORLD)
  return x
end

function synchronize!(x::ComponentArrays.ComponentArray; root_rank::Integer=0)
  d = MPIExtensions.bcast!(ComponentArrays.getdata(x), root_rank, MPI.COMM_WORLD)
  return ComponentArrays.ComponentArray(d, ComponentArrays.getaxes(x))
end

# Ideally these things should be Tuples and not arrays
function synchronize!(x::AbstractArray; root_rank::Integer=0)
  return synchronize!.(x; root_rank)
end

function synchronize!(l::Optimisers.Leaf; root_rank::Integer=0)
  Setfield.@set! l.state = synchronize!(l.state; root_rank)
  return l
end

function synchronize!(x::Number; root_rank::Integer=0)
  return MPIExtensions.bcast!([x], root_rank, MPI.COMM_WORLD)[1]
end

# If we don't know what to synchronize, we don't do it
# -- Symbols, Nothing, Missing, Val, etc.
synchronize!(x; root_rank::Integer=0) = x
