module FluxMPIComponentArraysExt

isdefined(Base, :get_extension) ? (using ComponentArrays) : (using ..ComponentArrays)
using FluxMPI, MPI

function FluxMPI.synchronize!(x::ComponentArray; root_rank::Integer=0)
  d = FluxMPI.bcast!(getdata(x), root_rank, MPI.COMM_WORLD)
  return ComponentArray(d, getaxes(x))
end

end
