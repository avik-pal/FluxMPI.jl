"""
    synchronize!(x; root_rank::Integer=0)

Synchronize `x` across all processes. Note: this function is not in-place for CuArrays when MPI is not CUDA aware.
"""
function synchronize!(ps::Union{NamedTuple, Tuple}; root_rank::Integer=0)
  length(ps) == 0 && return ps
  return fmap(x -> synchronize!(x; root_rank), ps)
end

function synchronize!(x::AbstractArray{T}; root_rank::Integer=0) where {T <: Number}
  return bcast!(x, root_rank, MPI.COMM_WORLD)
end

# Ideally these things should be Tuples and not arrays
function synchronize!(x::AbstractArray; root_rank::Integer=0)
  return synchronize!.(x; root_rank)
end

function synchronize!(l::Optimisers.Leaf; root_rank::Integer=0)
  @set! l.state = synchronize!(l.state; root_rank)
  return l
end

function synchronize!(x::Number; root_rank::Integer=0)
  return bcast!([x], root_rank, MPI.COMM_WORLD)[1]
end

# If we don't know what to synchronize, we don't do it
# -- Symbols, Nothing, Missing, Val, etc.
synchronize!(x; root_rank::Integer=0) = x
